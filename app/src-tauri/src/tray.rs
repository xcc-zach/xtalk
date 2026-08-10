//! Native system-tray controls for the background desktop process.

use std::sync::Arc;

use tauri::{
    menu::{Menu, MenuItem},
    tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent},
    App, Manager,
};

use crate::{request_app_exit, wake_word::WakeWordSupervisor};

const OPEN_MENU_ID: &str = "open";
const DISABLE_WAKE_WORD_MENU_ID: &str = "disable-wake-word";
const QUIT_MENU_ID: &str = "quit";

/// Creates the App tray icon and its background-lifecycle menu.
pub(crate) fn setup(app: &mut App) -> tauri::Result<()> {
    let open = MenuItem::with_id(app, OPEN_MENU_ID, "打开 XTalk", true, None::<&str>)?;
    let disable_wake_word = MenuItem::with_id(
        app,
        DISABLE_WAKE_WORD_MENU_ID,
        "关闭语音唤醒",
        true,
        None::<&str>,
    )?;
    let quit = MenuItem::with_id(app, QUIT_MENU_ID, "退出", true, None::<&str>)?;
    let menu = Menu::with_items(app, &[&open, &disable_wake_word, &quit])?;
    let mut builder = TrayIconBuilder::with_id("main-tray")
        .menu(&menu)
        .show_menu_on_left_click(false)
        .on_menu_event(|app, event| match event.id.as_ref() {
            OPEN_MENU_ID => show_main_window(app),
            DISABLE_WAKE_WORD_MENU_ID => {
                let supervisor = app.state::<Arc<WakeWordSupervisor>>().inner().clone();
                let app_handle = app.clone();
                tauri::async_runtime::spawn(async move {
                    if let Err(error) = supervisor.set_enabled(&app_handle, false, false).await {
                        eprintln!("could not update wake-word detection: {error}");
                    }
                });
            }
            QUIT_MENU_ID => request_app_exit(app.clone()),
            _ => {}
        })
        .on_tray_icon_event(|tray, event| {
            if let TrayIconEvent::Click {
                button: MouseButton::Left,
                button_state: MouseButtonState::Up,
                ..
            } = event
            {
                show_main_window(tray.app_handle());
            }
        });
    if let Some(icon) = app.default_window_icon() {
        builder = builder.icon(icon.clone());
    }
    builder.build(app)?;
    Ok(())
}

fn show_main_window(app: &tauri::AppHandle) {
    if let Some(window) = app.get_webview_window("main") {
        let _ = window.unminimize();
        let _ = window.show();
        let _ = window.set_focus();
    }
}
