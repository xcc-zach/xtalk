//! Text preparation shared by the native MOSS ONNX service endpoints.

const SENTENCE_ENDINGS: &[char] = &['.', '!', '?', '。', '！', '？'];
const CLAUSE_BOUNDARIES: &[char] = &[',', '，', '、', ';', '；', ':', '：'];

/// Convert mixed Chinese text and absolute filesystem paths into stable TTS input.
pub(crate) fn normalize_for_speech(text: &str) -> String {
    let collapsed = text.split_whitespace().collect::<Vec<_>>().join(" ");
    let mut normalized = replace_absolute_paths(&collapsed);
    for (from, to) in [
        ("，路径是 slash", "。路径如下。slash"),
        (",路径是 slash", "。路径如下。slash"),
        ("路径是 slash", "路径如下。slash"),
        ("。，", "。"),
        ("。,", "。"),
        (".，", ". "),
        (".,", ". "),
        ("。。", "。"),
        ("..", "."),
    ] {
        normalized = normalized.replace(from, to);
    }
    add_script_boundary_spacing(&normalized)
}

/// Split normalized text at sentence endings without losing their punctuation.
pub(crate) fn sentence_chunks(text: &str) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut current = String::new();
    for character in text.chars() {
        current.push(character);
        if SENTENCE_ENDINGS.contains(&character) {
            let chunk = current.trim();
            if !chunk.is_empty() {
                chunks.push(chunk.to_owned());
            }
            current.clear();
        }
    }
    let tail = current.trim();
    if !tail.is_empty() {
        chunks.push(tail.to_owned());
    }
    chunks
}

/// Split one unstable long CJK sentence into closed clauses.
pub(crate) fn clause_chunks(text: &str) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut current = String::new();
    for character in text.chars() {
        current.push(character);
        if CLAUSE_BOUNDARIES.contains(&character) {
            push_closed_chunk(&mut chunks, &current);
            current.clear();
        }
    }
    push_closed_chunk(&mut chunks, &current);
    chunks
}

/// Return whether a sentence has enough clause boundaries to benefit from splitting.
pub(crate) fn clause_boundary_count(text: &str) -> usize {
    text.chars()
        .filter(|character| CLAUSE_BOUNDARIES.contains(character))
        .count()
}

/// Return whether text contains a CJK code point.
pub(crate) fn contains_cjk(text: &str) -> bool {
    text.chars().any(is_cjk)
}

fn replace_absolute_paths(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut output = String::with_capacity(text.len());
    let mut cursor = 0;
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] != b'/' || !path_can_start(text, index) {
            index += 1;
            continue;
        }
        let mut end = index;
        let mut slash_count = 0;
        while end < bytes.len() {
            let byte = bytes[end];
            if byte == b'/' {
                slash_count += 1;
                end += 1;
            } else if byte.is_ascii_alphanumeric()
                || matches!(byte, b'.' | b'_' | b'+' | b'~' | b'-')
            {
                end += 1;
            } else {
                break;
            }
        }
        if slash_count < 2 || end <= index + 2 {
            index += 1;
            continue;
        }
        output.push_str(&text[cursor..index]);
        output.push_str(&pronounce_path(&text[index..end]));
        output.push('.');
        cursor = end;
        index = end;
    }
    output.push_str(&text[cursor..]);
    output
}

fn path_can_start(text: &str, index: usize) -> bool {
    if index == 0 {
        return true;
    }
    text[..index].chars().next_back().is_some_and(|character| {
        character.is_whitespace()
            || matches!(
                character,
                '，' | '。' | '！' | '？' | '；' | '：' | ',' | ';'
            )
    })
}

fn pronounce_path(path: &str) -> String {
    path.split('/')
        .filter(|component| !component.is_empty())
        .map(|component| format!("slash {}", pronounce_path_component(component)))
        .collect::<Vec<_>>()
        .join(", ")
}

fn pronounce_path_component(component: &str) -> String {
    let chars = component.chars().collect::<Vec<_>>();
    let mut expanded = String::new();
    for (index, character) in chars.iter().copied().enumerate() {
        let previous = index
            .checked_sub(1)
            .and_then(|value| chars.get(value))
            .copied();
        let next = chars.get(index + 1).copied();
        let camel_boundary = previous.is_some_and(|value| {
            (value.is_ascii_lowercase() || value.is_ascii_digit()) && character.is_ascii_uppercase()
        }) || (previous.is_some_and(|value| value.is_ascii_uppercase())
            && character.is_ascii_uppercase()
            && next.is_some_and(|value| value.is_ascii_lowercase()));
        if camel_boundary && !expanded.ends_with(' ') {
            expanded.push(' ');
        }
        match character {
            '.' => expanded.push_str(" dot "),
            '-' | '_' => expanded.push(' '),
            _ => expanded.push(character),
        }
    }
    expanded
        .split_whitespace()
        .flat_map(|token| {
            if token.eq_ignore_ascii_case("xtalk") {
                vec!["X".to_owned(), "Talk".to_owned()]
            } else if (2..=6).contains(&token.len())
                && token.bytes().all(|byte| byte.is_ascii_uppercase())
            {
                token
                    .chars()
                    .map(|character| character.to_string())
                    .collect()
            } else {
                vec![token.to_owned()]
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn add_script_boundary_spacing(text: &str) -> String {
    let characters = text.chars().collect::<Vec<_>>();
    let mut output = String::with_capacity(text.len());
    for (index, character) in characters.iter().copied().enumerate() {
        output.push(character);
        let Some(next) = characters.get(index + 1).copied() else {
            continue;
        };
        if !character.is_whitespace()
            && !next.is_whitespace()
            && ((is_cjk(character) && next.is_ascii_alphanumeric())
                || (character.is_ascii_alphanumeric() && is_cjk(next)))
        {
            output.push(' ');
        }
    }
    output
}

fn push_closed_chunk(chunks: &mut Vec<String>, raw: &str) {
    let trimmed = raw.trim_matches(|character: char| {
        character.is_whitespace() || CLAUSE_BOUNDARIES.contains(&character)
    });
    if trimmed.is_empty() {
        return;
    }
    let mut chunk = trimmed.to_owned();
    if !chunk
        .chars()
        .last()
        .is_some_and(|character| SENTENCE_ENDINGS.contains(&character))
    {
        chunk.push(if contains_cjk(&chunk) { '。' } else { '.' });
    }
    chunks.push(chunk);
}

fn is_cjk(character: char) -> bool {
    matches!(
        character as u32,
        0x3400..=0x4DBF | 0x4E00..=0x9FFF | 0x3040..=0x30FF | 0xAC00..=0xD7AF
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prepares_mixed_text_and_paths_for_speech() {
        let text =
            "文件路径是 /Applications/XTalk.app/Contents/MacOS/xtalk-desktop，看来它装好了。";
        assert_eq!(
            normalize_for_speech(text),
            "文件路径如下。slash Applications, slash X Talk dot app, slash Contents, slash Mac O S, slash X Talk desktop. 看来它装好了。"
        );
    }

    #[test]
    fn closes_long_chinese_clause_chunks() {
        assert_eq!(
            clause_chunks("我是你的智能助手，随时准备帮你解答问题或处理任务，咱们开始吧。"),
            [
                "我是你的智能助手。",
                "随时准备帮你解答问题或处理任务。",
                "咱们开始吧。",
            ]
        );
    }
}
