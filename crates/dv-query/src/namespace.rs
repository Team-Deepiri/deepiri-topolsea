//! Multi-tenant namespace helpers (C13).

/// Default namespace when none is specified.
pub const DEFAULT_NAMESPACE: &str = "_default";

/// Qualify a collection name with a namespace → on-disk path segment.
///
/// `_default/foo` stores as `foo` for backward compatibility.
/// Other namespaces store as `{ns}/{name}`.
pub fn qualify_collection(namespace: &str, name: &str) -> String {
    let ns = normalize_namespace(namespace);
    let name = name.trim_matches('/');
    if ns == DEFAULT_NAMESPACE {
        name.to_string()
    } else {
        format!("{ns}/{name}")
    }
}

pub fn normalize_namespace(ns: &str) -> String {
    let ns = ns.trim().trim_matches('/');
    if ns.is_empty() {
        DEFAULT_NAMESPACE.to_string()
    } else {
        ns.to_string()
    }
}

/// Strip namespace prefix from a stored collection name for listing.
pub fn strip_namespace<'a>(namespace: &str, qualified: &'a str) -> Option<&'a str> {
    let ns = normalize_namespace(namespace);
    if ns == DEFAULT_NAMESPACE {
        if qualified.contains('/') {
            return None;
        }
        return Some(qualified);
    }
    let prefix = format!("{ns}/");
    qualified.strip_prefix(&prefix)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_keeps_flat_name() {
        assert_eq!(qualify_collection("_default", "demo"), "demo");
        assert_eq!(qualify_collection("", "demo"), "demo");
    }

    #[test]
    fn tenant_prefixes() {
        assert_eq!(qualify_collection("acme", "demo"), "acme/demo");
    }
}
