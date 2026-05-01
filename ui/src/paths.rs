//! Cross-platform appdata directory helpers.
//!
//! Models are persisted under a per-user data directory; users no longer need
//! to type a path.

use std::env;
use std::fs;
use std::io;
use std::path::PathBuf;

/// Root directory: `<platform-appdata>/neuralcabin/`. Created on demand.
pub fn appdata_dir() -> io::Result<PathBuf> {
    let dir = if cfg!(target_os = "windows") {
        env::var_os("APPDATA")
            .map(PathBuf::from)
            .or_else(|| {
                env::var_os("USERPROFILE")
                    .map(|p| PathBuf::from(p).join("AppData").join("Roaming"))
            })
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "no APPDATA / USERPROFILE"))?
    } else if cfg!(target_os = "macos") {
        let home = env::var_os("HOME")
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "no HOME"))?;
        PathBuf::from(home).join("Library").join("Application Support")
    } else {
        if let Some(xdg) = env::var_os("XDG_DATA_HOME") {
            PathBuf::from(xdg)
        } else {
            let home = env::var_os("HOME")
                .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "no HOME"))?;
            PathBuf::from(home).join(".local").join("share")
        }
    };
    let dir = dir.join("neuralcabin");
    fs::create_dir_all(&dir)?;
    Ok(dir)
}

/// `<appdata>/networks/`. Created on demand.
pub fn networks_dir() -> io::Result<PathBuf> {
    let dir = appdata_dir()?.join("networks");
    fs::create_dir_all(&dir)?;
    Ok(dir)
}

/// Sanitise a freeform network name into a safe filename stem.
pub fn sanitize_filename(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    for c in name.chars() {
        if c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.' {
            out.push(c);
        } else if c.is_whitespace() {
            out.push('-');
        }
    }
    while out.starts_with('.') { out.remove(0); }
    if out.is_empty() { out.push_str("network"); }
    out
}

/// List all `.json` networks under `networks_dir()`, sorted alphabetically.
pub fn list_saved_networks() -> io::Result<Vec<(String, PathBuf)>> {
    let dir = networks_dir()?;
    let mut out = Vec::new();
    for entry in fs::read_dir(&dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) == Some("json") {
            if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                if !stem.is_empty() {
                    out.push((stem.to_string(), path));
                }
            }
        }
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(out)
}

/// Delete the on-disk file for a saved network, by stem.
#[allow(dead_code)]
pub fn delete_saved(stem: &str) -> io::Result<()> {
    let path = networks_dir()?.join(format!("{stem}.json"));
    if path.exists() { fs::remove_file(path)?; }
    Ok(())
}
