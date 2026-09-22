if (!globalThis.mindLensContentScriptLoaded) {
    globalThis.mindLensContentScriptLoaded = true;

    // This listener responds only when the popup asks for the current selection.
    // It does not scan the page or send page contents automatically.
    chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
        if (message?.type !== "GET_SELECTED_TEXT") {
            return;
        }

        const selectedText = window.getSelection()?.toString().trim() || "";
        sendResponse({ text: selectedText });
    });
}
