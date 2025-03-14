#pragma once
//
// Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto. Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.
//

//
// This module allows for a Korg nanoKontrol 2 USB MIDI controller to be used to tweak in-game variables.
// The Korg nanoKontrol 2 is a low cost device with buttons and sliders.  In many cases, tweaking using
// this device can be much more direct and much easier than using ImGUI.
// https://www.korg.com/uk/products/computergear/nanokontrol2/
//
// There are Init, Shutdown and Update calls that need making to initialise the system,
// shut it down, and update it (just call update once per frame).
//
// Controls are grouped into pages.  There are 4 pages numbered 0-3.  The current page can be
// selected using the << and >> buttons as a two bit binary number.  They will illuminate to
// show the current selected page.
//
// There are a couple of helper macros to handle the most common use cases of toggle
// buttons to control and react to a pre-existing bool, and sliders to control a pre-existing float.
//
// Declare a toggle button like this....
//     KORGI_TOGGLE( Variable, Page, Control )
//     E.g. to have a cheese toggle button on button S1, you would do this
//         KORGI_TOGGLE( g_cheeseEnable, 0, S1 )
//
// Declare a knob like this...
//     KORGI_KNOB( Variable, Page, Control, MinValue = 0, MaxValue = 1 )
//     E.g. to modulate your turbo encabulator on Slider 1, you would do this
//         KORGI_KNOB( g_turboEncabulator, 0, Slider1 )
//
// Another use of buttons is that of 'momentary' actions, where you want to test
// if a button has been pressed and perform some action.
// In order to use this mode, you need to create a korgi::Button variable that you can
// then call `WasMomentarilyPressed()` on to test if that button has just been pressed.
//
// g_launchMissilesButton = korgi::Button( 0, korgi::Control::M1, korgi::ButtonMode::Momentary );
//
// if( g_launchMissilesButton.WasMomentarilyPressed() )
// {
//     if( GentleConfirmationDialogue( "Are you sure?" ) )
// ...
//

#ifdef KORGI_ENABLED
//  External control
#   if (KORGI_ENABLED != 0) && (KORGI_ENABLED != 1)
#      error "If you define KORGI_ENABLED, please set it to 0 or 1"
#   endif
#else
//  Enable it by default, otherwise why would you be including it
#   define KORGI_ENABLED 1
#endif

// Ensure that korgi is only compiled into Windows platforms
#if KORGI_ENABLED && !defined(_WIN32)
    #undef KORGI_ENABLED
    #define KORGI_ENABLED 0
#endif

namespace korgi
{

#if KORGI_ENABLED
void Init();
void Shutdown();
void Update();
#else
static inline void Init() {}
static inline void Shutdown() {}
static inline void Update() {}
#endif

// Macro concatenation machinary
#define KORGI_TOKEN_PASTE(x, y) x##y
#define KORGI_CAT(x,y) KORGI_TOKEN_PASTE(x,y)

// Helpers for the easy cases to control existing bools, ints and floats
#if KORGI_ENABLED
#define KORGI_TOGGLE(variable, page, control) static korgi::Button KORGI_CAT(s_KorgButton_, __LINE__) (page, korgi::Control::##control, korgi::ButtonMode::BoolToggle, &( variable ));
#define KORGI_INT_TOGGLE(variable, page, control, offValue, onValue) static korgi::Button KORGI_CAT(s_KorgButton_, __LINE__) (page, korgi::Control::##control, (int*) &( variable ), int(offValue), int(onValue));
#define KORGI_KNOB(variable, page, control, ...) static korgi::Knob KORGI_CAT(s_KorgKnob_, __LINE__) (page, korgi::Control::##control, &( variable ), ##__VA_ARGS__ );
#else
#define KORGI_TOGGLE(...)
#define KORGI_INT_TOGGLE(...)
#define KORGI_KNOB(...)
#endif

enum class ButtonMode
{
    Momentary, // Use Button::wasMomentarilyPressed() to Get a single 'true' for each press.
    BoolToggle,
    IntToggle
};

// Enum for all the control channels on the Korg nanoKONTROL2
// Numbering starts from 1 to match the Confluence page descriptions.
enum class Control : unsigned char
{
    // 'S' Buttons
    S1 = 32,
    S2 = 33,
    S3 = 34,
    S4 = 35,
    S5 = 36,
    S6 = 37,
    S7 = 38,
    S8 = 39,

    // 'M' Buttons
    M1 = 48,
    M2 = 49,
    M3 = 50,
    M4 = 51,
    M5 = 52,
    M6 = 53,
    M7 = 54,
    M8 = 55,

    // 'R' Buttons
    R1 = 64,
    R2 = 65,
    R3 = 66,
    R4 = 67,
    R5 = 68,
    R6 = 69,
    R7 = 70,
    R8 = 71,

    // Other buttons
    PreviousTrack = 58,
    NextTrack = 59,
    Cycle = 46,
    SetMarker = 60,
    PreviousMarker = 61,
    NextMarker = 62,
    Rewind = 43,
    FastForward = 44,
    Stop = 42,
    Play = 41,
    Record = 45,

    // Knobs
    Knob1 = 16,
    Knob2 = 17,
    Knob3 = 18,
    Knob4 = 19,
    Knob5 = 20,
    Knob6 = 21,
    Knob7 = 22,
    Knob8 = 23,

    // Sliders
    Slider1 = 0,
    Slider2 = 1,
    Slider3 = 2,
    Slider4 = 3,
    Slider5 = 4,
    Slider6 = 5,
    Slider7 = 6,
    Slider8 = 7,
};

struct Button
{
#if KORGI_ENABLED
    // Constructor for controlling a bool variable
    Button(int page, Control controlChannel, ButtonMode mode, bool* pValue = nullptr);
    // Constructor for toggling an int variable between two states
    Button(int page, Control controlChannel, int* pValue, int offValue = 0, int onValue = 1);

    // Returns true only once as the button is pressed
    // (will not continue to return true as the button is held)
    bool WasMomentarilyPressed();

private:
    friend struct Controller;

    bool GetState() const;
    void SetState(bool state);
    bool GetLedStatus() const { return m_LedStatus;  }
    void SetLedStatus(bool status) { m_LedStatus = status; }
    int GetPage() const { return m_Page; }
    ButtonMode GetMode() const { return m_Mode; }

    ButtonMode m_Mode;
    void* m_pValue;
    int m_OffValue;
    int m_OnValue;
    bool m_LocalState; // If pValue initialised as nullptr
    bool m_PreviousState;
    bool m_LedStatus;
    int m_Page;
#else
    Button(int, Control, ButtonMode, bool* pValue = nullptr)
    {
        (void)pValue;
    }
    bool WasMomentarilyPressed() { return false; }
#endif
};

struct Knob
{
#if KORGI_ENABLED
    Knob(int page, Control controlChannel, float* pValue, float mi = 0.0f, float ma = 1.0f);

private:
    friend struct Controller;

    void SetValue(const float newRawValue)
    {
        *m_pValue = m_MinValue * (1.f - newRawValue) + m_MaxValue * newRawValue;
    }
    int GetPage() const { return m_Page; }

    float* m_pValue;
    float m_MinValue;
    float m_MaxValue;
    int m_Page;
#else
    Knob(int page, Control, float*, float mi = 0.0f, float ma = 1.0f)
    {
        (void)mi; (void)ma;
    }
#endif
};

} // namespace korgi
