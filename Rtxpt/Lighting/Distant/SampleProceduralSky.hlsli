/*
* Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#ifndef __SAMPLE_PROCEDURAL_SKY_HLSLI__
#define __SAMPLE_PROCEDURAL_SKY_HLSLI__

#include "precomputed_sky.hlsli"

#include "../../PathTracer/Utils/Utils.hlsli"

struct ProceduralSkyConstants
{
    AtmosphereParameters    SkyParams;

    float3                  FinalRadianceMultiplier;
    float                   _padding0;

    float3                  SunDir;
    float                   CloudsTime;

    float3                  GroundAlbedo;
    float                   SunAngularRadius;

    // precomputed for performance
    float                   sun_tan_half_angle;
    float                   sun_cos_half_angle;
    float                   sun_solid_angle;
    float                   _padding2;
    float3                  physical_sky_ground_radiance;
    float                   cloud_density_offset;
   
    // used for clouds primarily
	float                   sky_transmittance;
	float                   sky_phase_g;
	float                   sky_amb_phase_g;
	float                   sky_scattering;
};

// Note: sky/clouds code taken from https://github.com/NVIDIA/Q2RTX/blob/master/src/refresh/vkpt/shader/physical_sky.comp 
// Original license below.
/*
Copyright (C) 2019, NVIDIA CORPORATION. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with this program; if not, write to the Free Software Foundation, Inc.,
51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
*/

#include "precomputed_sky.hlsli"

// The rest below is shader only code
#if !defined(__cplusplus) || defined(__INTELLISENSE__) 

#include "../../PathTracer/Utils/NoiseAndSequences.hlsli"

struct ProceduralSkyWorkingContext
{
    ProceduralSkyConstants      Consts;
    SamplerState                SamplerLinearWrap;
    Texture2D                   TransmittanceTexture;
    Texture3D                   ScatterTexture;
    Texture2D                   IrradianceTexture;
    Texture3D                   CloudsTexture;
    Texture2D                   NoiseTexture;
};

// Q2RTX constants/properties
bool drawClouds()       { return true; }

#define SKY_PI 3.1415926535897932384626433832795

static const float CLOUD_START = 2.0;	// A height over the Earth surface, where the cloud layer starts, in km
static const float CLOUD_HEIGHT = 1.4;	// The cloud layer thickness, in km

#define CLOUDS_FINE_COUNT       24
#define CLOUDS_SKY_SUN_COUNT    1

#define SUN_RAY_LENGTH          (CLOUD_HEIGHT / (CLOUDS_SKY_SUN_COUNT * 4))

#define HORIZONFADE             0.2


// returns distance to sphere intersection, where:
// 	origin - intersecting ray origin point
// 	dir - intersecting ray direction, normalized
//     spherePos - center of the sphere
//     sphereRad - sphere radius
float intersectSphere(float3 origin, float3 dir, float3 spherePos, float sphereRad) 
{
    float3 oc = origin - spherePos;
    float b = 2.0 * dot(dir, oc);
    float c = dot(oc, oc) - sphereRad*sphereRad;
    float disc = b * b - 4.0 * c;
    if (disc < 0.0)
        return -1.0;    
    float q = (-b + ((b < 0.0) ? -sqrt(disc) : sqrt(disc))) / 2.0;
    float t0 = q;
    float t1 = c / q;
    if (t0 > t1) 
	{
        float temp = t0;
        t0 = t1;
        t1 = temp;
    }
    if (t1 < 0.0)
        return -1.0;
    return (t0 < 0.0) ? t1 : t0;
}

//  Combine 3D cloud texture layers to get the cloud density at the given point
float GetDensity(ProceduralSkyWorkingContext workingContext, float3 step, float raylen, float current_ray)
{
    float w = saturate(current_ray / raylen);
    float sizeScale = 0.5;
    float3 uvw1 = float3(step.xy * 0.1 * sizeScale, w);
    float3 uvw2 = float3(step.xy * sizeScale, w);

    float2 offset = workingContext.Consts.CloudsTime * float2( 0.707, 0.707 ) * 0.01;
    float4 cloud1 = workingContext.CloudsTexture.SampleLevel( workingContext.SamplerLinearWrap, uvw1 + float3( offset, 0 ), 0 );
    float4 cloud2 = workingContext.CloudsTexture.SampleLevel( workingContext.SamplerLinearWrap, uvw2 + float3( offset, 0 ), 0 );

    return cloud1.r + (cloud2.g - 0.5) * 0.1;
    return cloud1.r;
}

// Mie phase function
float HenyeyGreenstein(float mu, float inG) {
    return (1.-inG * inG)/(pow(1.+inG*inG - 2.0 * inG*mu, 1.5)*4.0* SKY_PI);
}

// Calculates transmittance to the given direction from the given point
float ComputeSunTransmittanceAtPos(ProceduralSkyWorkingContext workingContext, float3 camera, float3 view_dir, float raylen, float current_ray, float raypart, int stepCount)
{
    float3 step = camera;
    float3 delta = view_dir * raypart;
    float cray = current_ray;
    float Transmittance = 1.0f;
    for (int i=0; i<stepCount; i++)
    {
        float Density = GetDensity(workingContext, step, raylen, cray);
        Density = max(0, Density - workingContext.Consts.cloud_density_offset) / (1.001f - workingContext.Consts.cloud_density_offset);
        if (Density > 0.001)
        {
            Transmittance *= exp(-workingContext.Consts.sky_transmittance * raypart * Density);
        }
        cray += raypart;
        step += delta;
        if (cray > raylen)
            break;
    }
    return clamp(Transmittance, 0, 1);
}

// Raymarches from the camera point in the view_dir direction to the raylen distance
float4 FineRaymarching(ProceduralSkyWorkingContext workingContext, float3 camera, float3 view_dir, float raylen, float current_ray, float raypart, int stepCount/*, out float3 spoint*/, inout uint randHash)
{
    ProceduralSkyConstants procSkyConsts = workingContext.Consts;
    AtmosphereParameters skyParams = procSkyConsts.SkyParams;
    const float rndScale = 0.3; float rndSample = (Hash32ToFloat( randHash ) * rndScale - rndScale*0.5);
    float cray = current_ray + rndSample * raypart;
    float3 delta = view_dir * raypart;
    float3 step = camera + rndSample * delta;
    
    // Just getting the transmittance from the sun to the point
    float3 sun_transmittance;
    float3 radiance = GetSkyRadiance(procSkyConsts.SkyParams, workingContext.TransmittanceTexture, workingContext.SamplerLinearWrap, workingContext.ScatterTexture, workingContext.SamplerLinearWrap, camera, procSkyConsts.SunDir, procSkyConsts.SunDir, sun_transmittance);
    float3 sun_direct_radiance = sun_transmittance;
    
    float3 sky_irradiance = GetSkyIrradiance(procSkyConsts.SkyParams, workingContext.TransmittanceTexture, workingContext.SamplerLinearWrap, workingContext.IrradianceTexture, workingContext.SamplerLinearWrap, camera, float3(0,0,0), procSkyConsts.SunDir);
    
    float PhaseFunc = HenyeyGreenstein(dot(procSkyConsts.SunDir, view_dir), procSkyConsts.sky_phase_g);
    float AmbientPhaseFunc = HenyeyGreenstein(dot(procSkyConsts.SunDir, view_dir), procSkyConsts.sky_amb_phase_g);
    float Transmittance = 1.0f;
    float3 Scattering = float3(0,0,0);

    const float stepSize = 1.0 / float(stepCount);

    for (int i=0; i<stepCount; i++)
    {
        float Density = GetDensity(workingContext, step, raylen, cray);
        
        Density = max(0, Density - workingContext.Consts.cloud_density_offset) / (1.001f - workingContext.Consts.cloud_density_offset);

        const float fadeRange = 0.04;
        float fade = saturate( (Density /*- fadeRange*/) / fadeRange );
        if (fade>0)
        {
            // if (spoint.x == 0 && spoint.y == 0 && spoint.z == 0)
            //     spoint = step;
            Transmittance *= exp(-procSkyConsts.sky_transmittance * raypart * Density * fade);
            
            float SunTransmittance = ComputeSunTransmittanceAtPos(workingContext, step, procSkyConsts.SunDir, raylen, cray, SUN_RAY_LENGTH, CLOUDS_SKY_SUN_COUNT);

            float3 S = procSkyConsts.sky_scattering * stepSize * (PhaseFunc * sun_direct_radiance * SunTransmittance + AmbientPhaseFunc * sky_irradiance);
            Scattering += S * Transmittance;
        }
        
        cray += raypart;
        step += delta;
        if (cray > raylen)
            break;
    }

    return float4(Scattering, Transmittance);
}

float4 RaymarchClouds(ProceduralSkyWorkingContext workingContext, float3 camera, float3 view_dir, float distToAtmStart, float distToAtmEnd/*, out float3 spoint*/, inout uint randHash)
{
    float current_ray = 0;
    float raylen = distToAtmEnd - distToAtmStart;

    return FineRaymarching(workingContext, camera + distToAtmStart * view_dir, view_dir, raylen, current_ray, raylen / CLOUDS_FINE_COUNT, CLOUDS_FINE_COUNT/*, spoint*/, randHash);
}

float4 ProceduralSkyLowRes( uint cubeDim, uint3 cubePosFace, const float3 viewDirection, ProceduralSkyWorkingContext workingContext )
{
    // if (dot(viewDirection, float3(0, 1, 0)) > 0.99 ) return float4(0,1,0,1);
    // if (dot(viewDirection, float3(0, 0, 1)) > 0.99 ) return float4(0,0,1,1);
    // if (dot(viewDirection, float3(1, 0, 0)) > 0.99 ) return float4(1,0,0,1);

    uint randHash = Hash32Combine( Hash32Combine(Hash32(cubePosFace.x), cubePosFace.y), cubePosFace.z );
    
    float3 eyeVec = viewDirection;
 
    AtmosphereParameters skyParams = workingContext.Consts.SkyParams;

    float3 camera = float3(0,0,6360.1);
    float3 sun_transmittance = float3(0,0,0);

    float3 radiance = GetSkyRadiance(skyParams, workingContext.TransmittanceTexture, workingContext.SamplerLinearWrap, workingContext.ScatterTexture, workingContext.SamplerLinearWrap, camera, eyeVec.xyz, workingContext.Consts.SunDir, sun_transmittance);
    float3 sun_direct_radiance = sun_transmittance;

    sun_direct_radiance /= workingContext.Consts.sun_solid_angle;
    sun_direct_radiance *= pow(clamp((dot(eyeVec, workingContext.Consts.SunDir) - workingContext.Consts.sun_cos_half_angle) * 1000 + 0.875, 0, 1), 10);
        
    radiance += sun_direct_radiance;
        
    float CloudsVisible = saturate( (dot(normalize(camera), eyeVec) + 0.05) * 10.0 );
    [branch] if (CloudsVisible > 0 && drawClouds()) 
    {
        const float ATM_START = 6360.1+CLOUD_START;
        const float ATM_END = ATM_START+CLOUD_HEIGHT;

        float distToAtmStart = intersectSphere(camera, normalize(eyeVec+float3(0,0,HORIZONFADE)), float3(0.0, 0.0, 0.0), ATM_START);
        float distToAtmEnd = intersectSphere(camera, normalize(eyeVec+float3(0,0,HORIZONFADE)), float3(0.0, 0.0, 0.0), ATM_END);

        float4 color = RaymarchClouds(workingContext, camera, eyeVec, distToAtmStart, distToAtmEnd, randHash);
        return color;
    }
    return float4(0,0,0,0);
}

float3 ProceduralSky( uint cubeDim, uint3 cubePosFace, const float3 viewDirection, ProceduralSkyWorkingContext workingContext, TextureCube<float4> lowResPrePassCube, float3 cubeDir, float3 cubeDirRight, float3 cubeDirBottom )
{
    uint randHash = Hash32Combine( Hash32Combine(Hash32(cubePosFace.x), cubePosFace.y), cubePosFace.z );

    float3 eyeVec = viewDirection;
 
    AtmosphereParameters skyParams = workingContext.Consts.SkyParams;

    float3 camera = float3(0,0,6360.1);
    float3 sun_transmittance = float3(0,0,0);

    float3 radiance = GetSkyRadiance(skyParams, workingContext.TransmittanceTexture, workingContext.SamplerLinearWrap, workingContext.ScatterTexture, workingContext.SamplerLinearWrap, camera, eyeVec.xyz, workingContext.Consts.SunDir, sun_transmittance);
    float3 sun_direct_radiance = sun_transmittance;

    sun_direct_radiance /= workingContext.Consts.sun_solid_angle;
    sun_direct_radiance *= pow(clamp((dot(eyeVec, workingContext.Consts.SunDir) - workingContext.Consts.sun_cos_half_angle) * 1000 + 0.875, 0, 1), 10);
        
    radiance += sun_direct_radiance;
        
    float CloudsVisible = saturate( (dot(normalize(camera), eyeVec) + 0.02) * 5.0 );
    [branch] if (CloudsVisible > 0 && drawClouds()) 
    {

        const float ATM_START = 6360.1+CLOUD_START;
        const float ATM_END = ATM_START+CLOUD_HEIGHT;

        float distToAtmStart = intersectSphere(camera, normalize(eyeVec+float3(0,0,HORIZONFADE)), float3(0.0, 0.0, 0.0), ATM_START);
        float distToAtmEnd = intersectSphere(camera, normalize(eyeVec+float3(0,0,HORIZONFADE)), float3(0.0, 0.0, 0.0), ATM_END);

        float3 spoint = camera + distToAtmStart * eyeVec;
        
    #if 0 // full res
        float4 color = RaymarchClouds(workingContext, camera, eyeVec, distToAtmStart, distToAtmEnd/*, spoint*/, randHash);
        //color.w = pow(color.w, 0.01);
    #elif 0 // half by half res, bilinear filter
        float4 color = lowResPrePassCube.SampleLevel( workingContext.SamplerLinearWrap, cubeDir, 0.0 );
    #else   // half by half res, 3x3 filter
        float pixSize = 20.0 / float(cubeDim);
        float4 color = float4(0,0,0,0);
        int counter = 0;
        int steps = 1; float scale = 2.1;
        for( int x = -steps; x <= steps; x++ )
            for( int y = -steps; y <= steps; y++ )
            {
                color += lowResPrePassCube.SampleLevel( workingContext.SamplerLinearWrap, normalize(cubeDir+scale*cubeDirRight*x+scale*cubeDirBottom*y), 0.0 );
                counter++;
            }
        color /= float(counter);
    #endif
        color.w = 1-((1-color.w)*CloudsVisible);

        float3 ground_transmittance = float3(0,0,0);
        float3 radiance_to_point = GetSkyRadianceToPoint(skyParams, workingContext.TransmittanceTexture, workingContext.SamplerLinearWrap, workingContext.ScatterTexture, workingContext.SamplerLinearWrap, camera, spoint, workingContext.Consts.SunDir, ground_transmittance);
        color.xyz = color.xyz * ground_transmittance + radiance_to_point * 0.9;

        radiance = lerp(color.xyz, radiance, color.w);
    }
    
    radiance *= workingContext.Consts.FinalRadianceMultiplier;

    return radiance;
}

#endif // #ifndef __cplusplus

#endif // #ifndef __SAMPLE_PROCEDURAL_SKY_HLSLI__