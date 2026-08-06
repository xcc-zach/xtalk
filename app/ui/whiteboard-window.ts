import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";

const SHOW_COMMAND = "show_whiteboard_window";
const HIDE_COMMAND = "hide_whiteboard_window";
const SET_VISIBLE_COMMAND = "set_whiteboard_window_visible";
const VISIBLE_COMMAND = "is_whiteboard_window_visible";

export const WHITEBOARD_HIDDEN_EVENT = "whiteboard-window-hidden";
export const WHITEBOARD_VISIBLE_STORAGE_KEY = "xtalk.whiteboard.visible";

/**
 * Opens the global whiteboard window, creating it on first use.
 *
 * @returns Promise resolving to the resulting window visibility.
 */
export function showWhiteboardWindow(): Promise<boolean> {
  return invoke<boolean>(SHOW_COMMAND);
}

/**
 * Hides the global whiteboard window without destroying it.
 *
 * @returns Promise resolving to whether a window existed to hide.
 */
export function hideWhiteboardWindow(): Promise<boolean> {
  return invoke<boolean>(HIDE_COMMAND);
}

/**
 * Sets the global whiteboard window visibility.
 *
 * @param visible Whether the window should be shown or hidden.
 * @returns Promise resolving to the resulting visibility.
 */
export function setWhiteboardWindowVisible(visible: boolean): Promise<boolean> {
  return invoke<boolean>(SET_VISIBLE_COMMAND, { visible });
}

/**
 * Queries whether the global whiteboard window is currently visible.
 *
 * @returns Promise resolving to the current window visibility.
 */
export function isWhiteboardWindowVisible(): Promise<boolean> {
  return invoke<boolean>(VISIBLE_COMMAND);
}

/**
 * Subscribes to the event emitted when the whiteboard window is hidden.
 *
 * @param listener Callback invoked after the window is hidden.
 * @returns Promise resolving to the unsubscribe function.
 */
export function listenWhiteboardWindowHidden(
  listener: () => void,
): Promise<() => void> {
  return listen<void>(WHITEBOARD_HIDDEN_EVENT, listener);
}

/**
 * Persists the whiteboard window visibility preference.
 *
 * @param visible Visibility to remember across app restarts.
 */
export function persistWhiteboardVisiblePreference(visible: boolean): void {
  if (visible) {
    localStorage.setItem(WHITEBOARD_VISIBLE_STORAGE_KEY, "1");
  } else {
    localStorage.removeItem(WHITEBOARD_VISIBLE_STORAGE_KEY);
  }
}
