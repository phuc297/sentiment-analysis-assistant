from rapidfuzz import process, fuzz
import re
ABBREV_MAP = {
    # Phủ định
    "k": "không",
    "ko": "không",
    "kh": "không",
    "k0": "không",
    "kg": "không",
    "kp": "không phải",
    "kbh": "không bao giờ",
    "kbiet": "không biết",
    "kb": "không biết",
    "ch": "chưa",
    "cx": "cũng",

    # Tích cực / tiêu cực nhanh
    "zui": "vui",
    "vuii": "vui",
    "vl": "vãi",
    "vch": "vãi chưởng",
    "vc": "vãi",
    "qua": "quá",
    "qa": "quá",
    "cc": "cục cứt",

    # Cảm xúc tiêu cực
    "bucminh": "bực mình",
    "metvl": "mệt vãi",
    "metmoi": "mệt mỏi",

    # Teencode phổ biến
    "bj": "bị",
    "bjo": "bây giờ",
    "mik": "mình",
    "mk": "mình",
    "bn": "bao nhiêu",
    "bjan": "bạn",
    "ik": "ì",
    "đc": "được",
    "dc": "được",
    "dk": "được",
    "đk": "đăng ký",
    "ad": "admin",
    "ib": "nhắn tin",
    "inb": "nhắn tin",
    "rep": "trả lời",
    "cmt": "bình luận",
    "j": "gì",


    # Rút gọn chat
    "ok": "đồng ý",
    "oke": "đồng ý",
    "oki": "đồng ý",
    "okela": "đồng ý",
    "thx": "cảm ơn",
    "tks": "cảm ơn",
    "thanks": "cảm ơn",
    "pls": "làm ơn",
    "plz": "làm ơn",

    # Hành vi / trạng thái
    "hnay": "hôm nay",
    "hqua": "hôm qua",
    "mai": "ngày mai",
    "tom": "ngày mai",
    "vs": "với",
    "w": "với",
    "ms": "mới",
    "trc": "trước",

    # Viết không dấu sang đúng dấu
    "bietk": "biết không",

    "wa": "quá"
}


def normalize_repeated_chars(text):
    return re.sub(r"(.)\1{2,}", r"\1", text)


def normalize_teencode_chars(token):
    # j -> i
    token = token.replace("j", "i")

    # z -> d/g/gi/r (rất phức tạp). Ở đây dùng mapping an toàn nhất: z -> d
    token = token.replace("z", "d")

    # w -> u hoặc qu. Ở đây dùng dạng an toàn: w -> u
    token = token.replace("w", "u")

    # f -> ph
    token = token.replace("f", "ph")

    return token


def fuzzy_match(token, mapping_keys, threshold=85):
    match, score, _ = process.extractOne(
        token, mapping_keys, scorer=fuzz.ratio)
    return match if score >= threshold else None


def normalize_abbrev(sentence):
    sentence = normalize_repeated_chars(sentence.lower())
    tokens = sentence.split()

    normalized = []

    for tok in tokens:
        original_tok = tok

        # 1. Nếu nằm trong từ điển → map trực tiếp
        if tok in ABBREV_MAP:
            normalized.append(ABBREV_MAP[tok])
            continue

        # 2. Nếu không, thử fuzzy match
        fm = fuzzy_match(tok, ABBREV_MAP.keys())
        if fm:
            normalized.append(ABBREV_MAP[fm])
            continue

        # 3. Nếu vẫn không, chuẩn hóa ký tự teencode (j,z,w,f)
        tok2 = normalize_teencode_chars(tok)

        # 3a. Kiểm tra lại sau chuẩn hóa
        if tok2 in ABBREV_MAP:
            normalized.append(ABBREV_MAP[tok2])
            continue

        fm2 = fuzzy_match(tok2, ABBREV_MAP.keys())
        if fm2:
            normalized.append(ABBREV_MAP[fm2])
            continue

        # 4. Không chuẩn hóa được → giữ nguyên
        normalized.append(original_tok)

    return " ".join(normalized)


if __name__ == "__main__":
    print(normalize_abbrev("mik k bt j zui waaaa"))
    print(normalize_abbrev("hnay zui qua"))
