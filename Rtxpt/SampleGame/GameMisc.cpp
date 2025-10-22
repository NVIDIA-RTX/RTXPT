/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "GameScene.h"
#include "GameMisc.h"

#include <donut/core/log.h>
#include <donut/core/json.h>
#include <donut/core/math/math.h>
#include <donut/app/Camera.h>
#include <cmath>

#include "../ExtendedScene.h"

#include "../Misc/Korgi.h"

#include <fstream>
#include <iostream>
#include <thread>

using namespace donut;
using namespace donut::math;
using namespace donut::app;
using namespace donut::vfs;
using namespace donut::engine;
using namespace donut::render;

using namespace game;

bool Pose::Read( const Json::Value & node )
{
    if (node.isNull())
        return false;

    const auto& translation = node["translation"];
    if (!translation.isNull())
        translation >> Translation;

    const auto& rotation = node["rotation"];
    if (!rotation.isNull())
    {
        double4 value = double4(0.0, 0.0, 0.0, 1.0);
        rotation >> value;
        Rotation = dm::dquat::fromXYZW(value);
    }
    else
    {
        const auto& euler = node["euler"];
        if (!euler.isNull())
        {
            double3 value = double3::zero();
            euler >> value;
            Rotation = rotationQuat(value);
        }
    }

    const auto& scaling = node["scaling"];
    if (!scaling.isNull())
        scaling >> Scaling;

    const auto& keytime = node["keytime"];
    if (!keytime.isNull())
        keytime >> KeyTime;
        
    return true;
}

Json::Value Pose::Write()
{
    Json::Value ret;
    ret["translation"] << Translation;
    ret["rotation"] << double4(Rotation.x, Rotation.y, Rotation.z, Rotation.w);
    if (!(Scaling.x==1&&Scaling.y==1&&Scaling.z==1))
        ret["scaling"] << Scaling;
    ret["keytime"] << KeyTime;
    return ret;
}

bool KeyframeAnimation::Read(const Json::Value& node)
{
    assert(node.isArray());
    KeyTimeMin = FLT_MAX;
    KeyTimeMax = -FLT_MAX;
    for( int i = 0; i < node.size(); i++)
    {
        Pose key;
        key.Read(node[i]);
        Keys.push_back(key);
        KeyTimeMin = min( KeyTimeMin, key.KeyTime );
        KeyTimeMax = max( KeyTimeMax, key.KeyTime );
    }
    std::sort(Keys.begin(), Keys.end(), [ ](const Pose & a, const Pose & b) { return a.KeyTime < b.KeyTime; });
    return true;
}

Json::Value KeyframeAnimation::Write()
{
    Json::Value nodesArray;
    for (int i = 0; i < Keys.size(); i++)
        nodesArray.append( Keys[i].Write() );
    return nodesArray;
}

void KeyframeAnimation::FromKeys(const std::vector<game::Pose> & keys)
{
    KeyTimeMin = FLT_MAX;
    KeyTimeMax = -FLT_MAX;
    for (int i = 0; i < keys.size(); i++)
    {
        Pose key = keys[i];
        Keys.push_back(key);
        KeyTimeMin = min(KeyTimeMin, key.KeyTime);
        KeyTimeMax = max(KeyTimeMax, key.KeyTime);
    }
    std::sort(Keys.begin(), Keys.end(), [ ](const Pose& a, const Pose& b) { return a.KeyTime < b.KeyTime; });
}

affine3 Pose::ToTransform() const
{
    dm::daffine3 transformD = dm::scaling(this->Scaling);
    transformD *= this->Rotation.toAffine();
    transformD *= dm::translation(this->Translation);
    return dm::affine3(transformD);
}

std::tuple<float3, float3, float3> Pose::GetPosDirUp() const
{
    auto rot = affine3(Rotation.toAffine()).m_linear;
    float3 dir = rot.row0;
    float3 up = rot.row1;
    return std::make_tuple(float3(Translation), dir, up);
}

void Pose::SetTransform(const affine3 & transform)
{
    dm::decomposeAffine(daffine3(transform), &Translation, &Rotation, &Scaling);
}

void Pose::SetTransformFromCamera(const float3 & pos, const float3 & dir, const float3 & up)
{
    float3 camRight = normalize(cross(dir, up));
    SetTransform(affine3(dir, up, camRight, pos));
}

static double3 my_lerp(double3 a, double3 b, double u) { return a + (b - a) * u; }

static Pose lerp(const Pose & a, const Pose & b, double k)
{
    Pose ret;
    ret.KeyTime = lerp( a.KeyTime, b.KeyTime, k );
    ret.Translation = my_lerp( a.Translation, b.Translation, k );
    ret.Rotation = slerp<double>(a.Rotation, b.Rotation, k);
    ret.Scaling = my_lerp( a.Scaling, b.Scaling, k );
    return ret;
}

static double3 CatmullRom(const double3& v0, const double3& v1, const double3& v2, const double3& v3, double s)
{
    double3 ret;
    ret.x = 0.5 * (2.0 * v1.x + (v2.x - v0.x) * s + (2.0 * v0.x - 5.0 * v1.x + 4.0 * v2.x - v3.x) * s * s + (v3.x - 3.0 * v2.x + 3.0 * v1.x - v0.x) * s * s * s);
    ret.y = 0.5 * (2.0 * v1.y + (v2.y - v0.y) * s + (2.0 * v0.y - 5.0 * v1.y + 4.0 * v2.y - v3.y) * s * s + (v3.y - 3.0 * v2.y + 3.0 * v1.y - v0.y) * s * s * s);
    ret.z = 0.5 * (2.0 * v1.z + (v2.z - v0.z) * s + (2.0 * v0.z - 5.0 * v1.z + 4.0 * v2.z - v3.z) * s * s + (v3.z - 3.0 * v2.z + 3.0 * v1.z - v0.z) * s * s * s);
    return ret;
}

// Spherical quadrangle interpolation.
static dquat Squad(const dquat& q1, const dquat& q2, const dquat& q3, const dquat& q4, double t)
{
    dquat temp1 = slerp<double>(q1, q4, t);
    dquat temp2 = slerp<double>(q2, q3, t);

    return slerp<double>(temp2, temp1, 2.0f * t * (1.0f - t));
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// adapted from https://github.com/orangeduck/Motion-Matching/blob/main/quat.h - not yet merged with Donut math, which would be useful in future.
// see https://theorangeduck.com/page/cubic-interpolation-quaternions, license:
// MIT License
// 
// Copyright(c) 2021 Daniel Holden
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files(the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and /or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions :
// 
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// 
static inline dquat quat_inv(dquat q)
{
    return dquat(-q.w, q.x, q.y, q.z);
}
static inline dquat quat_mul(dquat q, dquat p)
{
    return dquat(
        p.w * q.w - p.x * q.x - p.y * q.y - p.z * q.z,
        p.w * q.x + p.x * q.w - p.y * q.z + p.z * q.y,
        p.w * q.y + p.x * q.z + p.y * q.w - p.z * q.x,
        p.w * q.z - p.x * q.y + p.y * q.x + p.z * q.w);
}
static inline dquat quat_inv_mul(dquat q, dquat p)
{
    return quat_mul(quat_inv(q), p);
}
static inline dquat quat_mul_inv(dquat q, dquat p)
{
    return quat_mul(q, quat_inv(p));
}
static inline dquat quat_abs(dquat x)
{
    return x.w < 0.0 ? -x : x;
}
static inline double3 quat_log(dquat q, float eps = 1e-8f)
{
    double length = sqrtf(q.x * q.x + q.y * q.y + q.z * q.z);

    if (length < eps)
    {
        return double3(q.x, q.y, q.z);
    }
    else
    {
        double halfangle = acosf(dm::clamp(q.w, -1.0, 1.0));
        return halfangle * (double3(q.x, q.y, q.z) / length);
    }
}
static inline double3 quat_to_scaled_angle_axis(dquat q, double eps = 1e-8f)
{
    return 2.0 * quat_log(q, eps);
}
static inline double quat_length(dquat q)
{
    return sqrt(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z);
}
static inline dquat quat_normalize(dquat q, const double eps = 1e-8f)
{
    return q / (quat_length(q) + eps);
}
static inline dquat quat_exp(double3 v, float eps = 1e-8f)
{
    float halfangle = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);

    if (halfangle < eps)
    {
        return quat_normalize(dquat(1.0, v.x, v.y, v.z));
    }
    else
    {
        float c = cosf(halfangle);
        float s = sinf(halfangle) / halfangle;
        return dquat(c, s * v.x, s * v.y, s * v.z);
    }
}
static inline dquat quat_from_scaled_angle_axis(double3 v, float eps = 1e-8f)
{
    return quat_exp(v / 2.0, eps);
}
void quat_hermite( dquat& rot, double3& vel, float x, dquat r0, dquat r1, double3 v0, double3 v1 )
{
    double w1 = 3 * x * x - 2 * x * x * x;
    double w2 = x * x * x - 2 * x * x + x;
    double w3 = x * x * x - x * x;

    double q1 = 6 * x - 6 * x * x;
    double q2 = 3 * x * x - 4 * x + 1;
    double q3 = 3 * x * x - 2 * x;

    double3 r1_sub_r0 = quat_to_scaled_angle_axis(quat_abs(quat_mul_inv(r1, r0)));

    rot = quat_mul(quat_from_scaled_angle_axis(w1 * r1_sub_r0 + w2 * v0 + w3 * v1), r0);
    vel = q1 * r1_sub_r0 + q2 * v0 + q3 * v1;
}
void quat_catmull_rom( dquat & rot, double3 & vel, double x, dquat r0, dquat r1, dquat r2, dquat r3 )
{
    double3 r1_sub_r0 = quat_to_scaled_angle_axis(quat_abs(quat_mul_inv(r1, r0)));
    double3 r2_sub_r1 = quat_to_scaled_angle_axis(quat_abs(quat_mul_inv(r2, r1)));
    double3 r3_sub_r2 = quat_to_scaled_angle_axis(quat_abs(quat_mul_inv(r3, r2)));

    double3 v1 = (r1_sub_r0 + r2_sub_r1) / 2.0;
    double3 v2 = (r2_sub_r1 + r3_sub_r2) / 2.0;
    quat_hermite(rot, vel, x, r1, r2, v1, v2);
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Mix of what looks good - see https://theorangeduck.com/page/cubic-interpolation-quaternions "Appendix: Preventing Overshoot" for even better
static Pose NiceSpline(const Pose& v0, const Pose& v1, const Pose& v2, const Pose& v3, double k)
{
    Pose ret;
    ret.KeyTime = lerp(v1.KeyTime, v2.KeyTime, k);
    ret.Translation = CatmullRom(v0.Translation, v1.Translation, v2.Translation, v3.Translation, k);
    double3 dummy;
    quat_catmull_rom(ret.Rotation, dummy, k, v0.Rotation, v1.Rotation, v2.Rotation, v3.Rotation); 
    //ret.Rotation = slerp<double>(v1.Rotation, v2.Rotation, k); // Squad(v0.Rotation, v1.Rotation, v2.Rotation, v3.Rotation, k);
    ret.Scaling = CatmullRom(v0.Scaling, v1.Scaling, v2.Scaling, v3.Scaling, k);
    return ret;
}

bool KeyframeAnimation::GetAt(double time, bool wrap, Pose & outPose, float & outAnimTime)
{
    if (Keys.size()==0)
        return false;

    if (!wrap)
        time = clamp( time, KeyTimeMin, KeyTimeMax );
    else
    {
        time = fmod( abs(time-KeyTimeMin+KeyTimeMax-KeyTimeMin), KeyTimeMax-KeyTimeMin );
        time += KeyTimeMin;
    }
    outAnimTime = time;

    int ia = 0, ib = Keys.size()-1;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
    if (LastFound>0 && Keys[LastFound-1].KeyTime <= time && Keys[LastFound].KeyTime >= time ) 
    {
        ia = LastFound - 1;
        ib = LastFound;
    }
    else
    {
        int start = 1; 
        if (LastFound>0 && Keys[LastFound-1].KeyTime <= time && Keys[LastFound].KeyTime <= time )
            start = LastFound;
        LastFound = start-1;

        for( int i = start; i < Keys.size(); i++ ) // linear search, awful heh
        {
            if (Keys[i].KeyTime>time)
            {
                ia = i-1;
                ib = i;
                LastFound = i;
                break;
            }
        }
    }

    Pose a = Keys[ia], b = Keys[ib];

    double span = b.KeyTime - a.KeyTime + 1e-15; assert( span > 0 );
    double off = time - a.KeyTime; assert( off >= 0 );
    double lerpK = saturate(off / span);

#if 0 // simple lerp
    outPose = lerp(a, b, lerpK);
#else // nice spline
    int iL = (ia > 0) ? (ia - 1) : (ia);
    int iR = (ib < (Keys.size()-1)) ? (ib + 1) : (ib);
    Pose L = Keys[iL], R = Keys[iR];
    outPose = NiceSpline(L, a, b, R, lerpK);
#endif
    return true;
}

bool        LightController::Read(const Json::Value& node)
{
    if (node.isNull())
        return false;

    node["color"] >> Color;
    node["intensity"] >> Intensity;
    node["enabled"] >> Enabled;
    node["toggleOnUIClick"] >> ToggleOnUIClick;
    node["innerAngle"] >> InnerAngle;
    node["outerAngle"] >> OuterAngle;
    node["autoOffTime"]>> AutoOffTime;
    node["autoOnTime"] >> AutoOnTime;
    node["autoOnOffTimeOffset"] >> AutoOnOffTimeOffset;
    return true;
}

Json::Value LightController::Write()
{
    Json::Value ret;
    ret["color"] << Color;
    ret["intensity"] << Intensity;
    ret["enabled"] << Enabled;
    ret["toggleOnUIClick"] << ToggleOnUIClick;
    ret["innerAngle"] << InnerAngle;
    ret["outerAngle"] << OuterAngle;
    if (AutoOffTime != 0.0f)
        ret["autoOffTime"] << AutoOffTime;
    if (AutoOnTime != 0.0f)
        ret["autoOnTime"] << AutoOnTime;
    if (AutoOnOffTimeOffset != 0.0f)
        ret["autoOnOffTimeOffset"] << AutoOnOffTimeOffset;
    return ret;

}


