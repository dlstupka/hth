#Requires AutoHotkey v2.0
#SingleInstance Force

; ============================================================
; SETTINGS
; ============================================================

LEFT_CROP   := 0.32   ; remove 20% from the FamilySearch window's left side
RIGHT_CROP  := 0.04   ; remove 20% from the FamilySearch window's right side
TOP_CROP    := 0.03   ; remove 5% from the top
BOTTOM_CROP := 0.30   ; remove 1% from the bottom

PAGE_LOAD_DELAY := 4000   ; FamilySearch page-load delay in milliseconds

running := false
batchActive := false
targetIterations := 0
completedIterations := 0

familySearchHwnd := 0
wordHwnd := 0

; ============================================================
; F6 — REGISTER MICROSOFT WORD
; Activate Word first, then press F6.
; ============================================================

F6:: {
    global wordHwnd

    hwnd := WinExist("A")

    if (!hwnd) {
        MsgBox("Could not identify the active window.")
        return
    }

    processName := WinGetProcessName("ahk_id " hwnd)

    if (StrLower(processName) != "winword.exe") {
        MsgBox(
            "The active window is not Microsoft Word."
            . "`nActive program: " processName
        )
        return
    }

    wordHwnd := hwnd

    ShowStatus(
        "Word registered"
        . "`n" WinGetTitle("ahk_id " wordHwnd)
    )
}

; ============================================================
; F7 — REGISTER FAMILYSEARCH
; Activate the FamilySearch browser window first, then press F7.
; ============================================================

F7:: {
    global familySearchHwnd

    hwnd := WinExist("A")

    if (!hwnd) {
        MsgBox("Could not identify the active window.")
        return
    }

    processName := StrLower(WinGetProcessName("ahk_id " hwnd))

    ; Prevent Word from accidentally being registered as FamilySearch.
    if (processName = "winword.exe") {
        MsgBox(
            "Microsoft Word is active."
            . "`nActivate the FamilySearch browser window, then press F7."
        )
        return
    }

    familySearchHwnd := hwnd

    ShowStatus(
        "FamilySearch registered"
        . "`n" WinGetTitle("ahk_id " familySearchHwnd)
    )
}

; ============================================================
; F8 — SET NUMBER OF CAPTURES
; ============================================================

F8:: {
    global running, batchActive
    global targetIterations, completedIterations
    global familySearchHwnd, wordHwnd

    if (batchActive) {
        MsgBox("Stop the current batch before changing the count.")
        return
    }

    if (!wordHwnd || !WinExist("ahk_id " wordHwnd)) {
        MsgBox(
            "Word has not been registered."
            . "`nActivate Word and press F6."
        )
        return
    }

    if (!familySearchHwnd || !WinExist("ahk_id " familySearchHwnd)) {
        MsgBox(
            "FamilySearch has not been registered."
            . "`nActivate FamilySearch and press F7."
        )
        return
    }

    result := InputBox(
        "Enter the number of FamilySearch images to capture:",
        "Set Capture Count",
        "w370 h145",
        "3"
    )

    if (result.Result != "OK")
        return

    value := Trim(result.Value)

    if !RegExMatch(value, "^\d+$") || Integer(value) < 1 {
        MsgBox("Enter a whole number greater than zero.")
        return
    }

    targetIterations := Integer(value)
    completedIterations := 0
    running := false

    ; Return to FamilySearch after the InputBox closes.
    WinActivate("ahk_id " familySearchHwnd)

    if !WinWaitActive("ahk_id " familySearchHwnd, , 3) {
        MsgBox("Could not activate FamilySearch.")
        return
    }

    ShowStatus(
        "Ready: " targetIterations " captures"
        . "`nPress F10 to begin"
    )
}

; ============================================================
; F10 — START / PAUSE / RESUME
; ============================================================

F10:: {
    global running, batchActive, targetIterations

    if (targetIterations < 1) {
        MsgBox("Press F8 and enter the capture count first.")
        return
    }

    running := !running

    if (running) {
        ShowStatus(batchActive ? "Resuming" : "Starting")

        if (!batchActive)
            SetTimer(RunBatch, -50)
    } else {
        ShowStatus("Paused")
    }
}

; ============================================================
; F9 — STOP AND RESET
; ============================================================

F9:: {
    global running, batchActive, completedIterations

    running := false
    batchActive := false
    completedIterations := 0

    ShowStatus("Stopped and reset")
}

; ============================================================
; ESC — EXIT SCRIPT
; ============================================================

Esc::ExitApp

; ============================================================
; MAIN CAPTURE LOOP
; ============================================================

RunBatch() {
    global running, batchActive
    global targetIterations, completedIterations
    global PAGE_LOAD_DELAY
    global familySearchHwnd, wordHwnd

    batchActive := true

    while (completedIterations < targetIterations) {

        ; Pause without losing progress.
        while (!running && batchActive)
            Sleep(100)

        if (!batchActive)
            return

        current := completedIterations + 1

        ; Verify that both registered windows still exist.
        if (!WinExist("ahk_id " familySearchHwnd)) {
            StopWithError("The registered FamilySearch window no longer exists.")
            return
        }

        if (!WinExist("ahk_id " wordHwnd)) {
            StopWithError("The registered Word window no longer exists.")
            return
        }

        ShowStatus(
            "Capturing " current " of " targetIterations
            . "`nRemaining: "
            . (targetIterations - completedIterations)
        )

        ; Bring FamilySearch forward before capturing.
        WinActivate("ahk_id " familySearchHwnd)

        if !WinWaitActive("ahk_id " familySearchHwnd, , 3) {
            StopWithError("Could not activate FamilySearch.")
            return
        }

        Sleep(500)

        ; Capture a percentage-based rectangle within the actual
        ; FamilySearch window, even if it is on another monitor.
        if !CaptureFamilySearchWindowArea() {
            StopWithError("Capture failed at iteration " current ".")
            return
        }

        ; Bring Word forward.
        WinActivate("ahk_id " wordHwnd)

        if !WinWaitActive("ahk_id " wordHwnd, , 3) {
            StopWithError("Could not activate Microsoft Word.")
            return
        }

        ; Paste the bitmap and start a fresh Word page.
        Send("^v")
        Sleep(1200)

        Send("^{Enter}")
        Sleep(500)

        ; Return to FamilySearch and advance one image.
        WinActivate("ahk_id " familySearchHwnd)

        if !WinWaitActive("ahk_id " familySearchHwnd, , 3) {
            StopWithError("Could not return to FamilySearch.")
            return
        }

        Sleep(400)
        Send("{Right}")

        completedIterations += 1

        ; Wait for the next image to render.
        Sleep(PAGE_LOAD_DELAY)
    }

    running := false
    batchActive := false

    SoundBeep(1000, 250)
    SoundBeep(1300, 350)

    ShowStatus(
        "Complete"
        . "`n" completedIterations
        . " captures finished"
    )
}

; ============================================================
; CALCULATE CROP FROM THE FAMILYSEARCH WINDOW
; ============================================================

CaptureFamilySearchWindowArea() {
    global familySearchHwnd
    global LEFT_CROP, RIGHT_CROP
    global TOP_CROP, BOTTOM_CROP

    try {
        WinGetPos(
            &windowX,
            &windowY,
            &windowWidth,
            &windowHeight,
            "ahk_id " familySearchHwnd
        )
    }
    catch {
        return false
    }

    if (windowWidth <= 0 || windowHeight <= 0)
        return false

    x := windowX + Round(windowWidth * LEFT_CROP)
    y := windowY + Round(windowHeight * TOP_CROP)

    width := Round(
        windowWidth * (1 - LEFT_CROP - RIGHT_CROP)
    )

    height := Round(
        windowHeight * (1 - TOP_CROP - BOTTOM_CROP)
    )

    return CaptureRectangleToClipboard(x, y, width, height)
}

; ============================================================
; WINDOWS GDI RECTANGLE CAPTURE TO CLIPBOARD
; ============================================================

CaptureRectangleToClipboard(x, y, width, height) {
    static SRCCOPY    := 0x00CC0020
    static CAPTUREBLT := 0x40000000
    static CF_BITMAP  := 2

    if (width <= 0 || height <= 0)
        return false

    hdcScreen := 0
    hdcMemory := 0
    hBitmap := 0
    hOldBitmap := 0
    clipboardOpened := false
    clipboardOwnsBitmap := false

    try {
        ; Get the desktop device context.
        hdcScreen := DllCall(
            "User32\GetDC",
            "Ptr", 0,
            "Ptr"
        )

        if (!hdcScreen)
            throw Error("GetDC failed.")

        ; Create a compatible in-memory device context.
        hdcMemory := DllCall(
            "Gdi32\CreateCompatibleDC",
            "Ptr", hdcScreen,
            "Ptr"
        )

        if (!hdcMemory)
            throw Error("CreateCompatibleDC failed.")

        ; Create the destination bitmap.
        hBitmap := DllCall(
            "Gdi32\CreateCompatibleBitmap",
            "Ptr", hdcScreen,
            "Int", width,
            "Int", height,
            "Ptr"
        )

        if (!hBitmap)
            throw Error("CreateCompatibleBitmap failed.")

        ; Select the bitmap into the memory DC.
        hOldBitmap := DllCall(
            "Gdi32\SelectObject",
            "Ptr", hdcMemory,
            "Ptr", hBitmap,
            "Ptr"
        )

        if (!hOldBitmap)
            throw Error("SelectObject failed.")

        ; Copy the selected screen rectangle into the bitmap.
        success := DllCall(
            "Gdi32\BitBlt",
            "Ptr", hdcMemory,
            "Int", 0,
            "Int", 0,
            "Int", width,
            "Int", height,
            "Ptr", hdcScreen,
            "Int", x,
            "Int", y,
            "UInt", SRCCOPY | CAPTUREBLT,
            "Int"
        )

        if (!success)
            throw Error("BitBlt failed.")

        ; Restore the original bitmap in the memory DC.
        DllCall(
            "Gdi32\SelectObject",
            "Ptr", hdcMemory,
            "Ptr", hOldBitmap,
            "Ptr"
        )
        hOldBitmap := 0

        ; Retry opening the clipboard because Word or the browser
        ; can briefly have it locked.
        Loop 10 {
            clipboardOpened := DllCall(
                "User32\OpenClipboard",
                "Ptr", 0,
                "Int"
            )

            if (clipboardOpened)
                break

            Sleep(100)
        }

        if (!clipboardOpened)
            throw Error("OpenClipboard failed.")

        if !DllCall("User32\EmptyClipboard", "Int")
            throw Error("EmptyClipboard failed.")

        result := DllCall(
            "User32\SetClipboardData",
            "UInt", CF_BITMAP,
            "Ptr", hBitmap,
            "Ptr"
        )

        if (!result)
            throw Error("SetClipboardData failed.")

        ; Windows owns hBitmap after SetClipboardData succeeds.
        clipboardOwnsBitmap := true
        hBitmap := 0

        return true
    }
    catch as err {
        ToolTip("Capture error:`n" err.Message)
        SetTimer(() => ToolTip(), -3000)
        return false
    }
    finally {
        if (clipboardOpened)
            DllCall("User32\CloseClipboard")

        if (hOldBitmap && hdcMemory) {
            DllCall(
                "Gdi32\SelectObject",
                "Ptr", hdcMemory,
                "Ptr", hOldBitmap,
                "Ptr"
            )
        }

        if (hBitmap && !clipboardOwnsBitmap) {
            DllCall(
                "Gdi32\DeleteObject",
                "Ptr", hBitmap
            )
        }

        if (hdcMemory) {
            DllCall(
                "Gdi32\DeleteDC",
                "Ptr", hdcMemory
            )
        }

        if (hdcScreen) {
            DllCall(
                "User32\ReleaseDC",
                "Ptr", 0,
                "Ptr", hdcScreen
            )
        }
    }
}

; ============================================================
; ERROR HANDLING
; ============================================================

StopWithError(message) {
    global running, batchActive

    running := false
    batchActive := false

    SoundBeep(500, 600)
    MsgBox(message)
}

; ============================================================
; STATUS TOOLTIP
; ============================================================

ShowStatus(message) {
    ToolTip(message)
    SetTimer(() => ToolTip(), -1800)
}
