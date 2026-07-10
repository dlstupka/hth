#Requires AutoHotkey v2.0
#SingleInstance Force

; ============================================================
; SETTINGS
; ============================================================

LEFT_CROP   := 0.20   ; remove 20% from left
RIGHT_CROP  := 0.20   ; remove 20% from right
TOP_CROP    := 0.05   ; remove 5% from top
BOTTOM_CROP := 0.01   ; remove 1% from bottom

PAGE_LOAD_DELAY := 4000   ; FamilySearch page-load delay in milliseconds

running := false
batchActive := false
targetIterations := 0
completedIterations := 0

familySearchHwnd := 0
wordHwnd := 0

; ============================================================
; F8 — SET CAPTURE COUNT
; Press F8 while FamilySearch is the active window.
; ============================================================

F8:: {
    global running, batchActive
    global targetIterations, completedIterations
    global familySearchHwnd, wordHwnd

    if (batchActive) {
        MsgBox("Stop the current batch before changing the count.")
        return
    }

    ; Remember the active FamilySearch browser window.
    familySearchHwnd := WinExist("A")

    if (!familySearchHwnd) {
        MsgBox("Could not identify the active FamilySearch window.")
        return
    }

    ; Find Microsoft Word.
    wordHwnd := WinExist("ahk_exe WINWORD.EXE")

    if (!wordHwnd) {
        MsgBox("Microsoft Word is not open.")
        return
    }

    result := InputBox(
        "Enter the number of FamilySearch images to capture:",
        "Set Capture Count",
        "w370 h145",
        "10"
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

    ; Return focus to FamilySearch after the dialog closes.
    WinActivate("ahk_id " familySearchHwnd)

    if !WinWaitActive("ahk_id " familySearchHwnd, , 3) {
        MsgBox("Could not reactivate the FamilySearch window.")
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
        MsgBox("Press F8 first and enter the capture count.")
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

        ShowStatus(
            "Capturing " current " of " targetIterations
            . "`nRemaining: "
            . (targetIterations - completedIterations)
        )

        ; Activate FamilySearch explicitly.
        WinActivate("ahk_id " familySearchHwnd)

        if !WinWaitActive("ahk_id " familySearchHwnd, , 3) {
            StopWithError("Could not activate the FamilySearch window.")
            return
        }

        Sleep(300)

        ; Capture the configured screen rectangle to the clipboard.
        if !CaptureFamilySearchArea() {
            StopWithError("Screen capture failed at capture " current ".")
            return
        }

        ; Activate Word explicitly.
        WinActivate("ahk_id " wordHwnd)

        if !WinWaitActive("ahk_id " wordHwnd, , 3) {
            StopWithError("Could not activate Microsoft Word.")
            return
        }

        ; Paste the screenshot.
        Send("^v")
        Sleep(1200)

        ; Put the next screenshot on a fresh Word page.
        Send("^{Enter}")
        Sleep(500)

        ; Return to FamilySearch explicitly.
        WinActivate("ahk_id " familySearchHwnd)

        if !WinWaitActive("ahk_id " familySearchHwnd, , 3) {
            StopWithError("Could not return to FamilySearch.")
            return
        }

        Sleep(300)

        ; Advance to the next FamilySearch image.
        Send("{Right}")

        completedIterations += 1

        ; Wait for the next FamilySearch image to render.
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
; CALCULATE THE SCREEN CAPTURE RECTANGLE
; ============================================================

CaptureFamilySearchArea() {
    global LEFT_CROP, RIGHT_CROP
    global TOP_CROP, BOTTOM_CROP

    x := Round(A_ScreenWidth * LEFT_CROP)
    y := Round(A_ScreenHeight * TOP_CROP)

    width := Round(
        A_ScreenWidth * (1 - LEFT_CROP - RIGHT_CROP)
    )

    height := Round(
        A_ScreenHeight * (1 - TOP_CROP - BOTTOM_CROP)
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
        ; Get the desktop screen device context.
        hdcScreen := DllCall(
            "User32\GetDC",
            "Ptr", 0,
            "Ptr"
        )

        if (!hdcScreen)
            throw Error("GetDC failed.")

        ; Create a memory device context.
        hdcMemory := DllCall(
            "Gdi32\CreateCompatibleDC",
            "Ptr", hdcScreen,
            "Ptr"
        )

        if (!hdcMemory)
            throw Error("CreateCompatibleDC failed.")

        ; Create a bitmap for the selected rectangle.
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

        ; Copy the selected screen area into the bitmap.
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

        ; Retry opening the clipboard.
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

        clipboardResult := DllCall(
            "User32\SetClipboardData",
            "UInt", CF_BITMAP,
            "Ptr", hBitmap,
            "Ptr"
        )

        if (!clipboardResult)
            throw Error("SetClipboardData failed.")

        ; Windows owns the bitmap after SetClipboardData succeeds.
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