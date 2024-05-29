import numpy as np
initial_list = ['가', '까', '나', '다', '따', '라', '마', '바', '빠', '사', '싸', '아', '자', '짜', '차', '카', '타', '파', '하']
medial_list = ['아', '애', '야', '얘', '어', '에', '여', '예', '오', '와', '왜', '외', '요', '우', '워', '웨', '위', '유', '으', '의', '이']
final_list = ['으', '윽', '윾', '윿', '은', '읁', '읂', '읃', '을', '읅', '읆', '읇', '읈', '읉', '읊', '읋', '음', '읍', '읎', '읏', '읐', '응', '읒', '읓', '읔', '읕', '읖', '읗']

def _decompose_korean_char(char):
    # 한글 유니코드 시작: 44032, 끝: 55199
    if not 44032 <= ord(char) <= 55199:
        return char, char, char
    
    # 한글 자모 시작 유니코드
    initial = 44032  # '가'의 유니코드
    
    # 초성, 중성, 종성 계산
    code = ord(char) - initial
    initial_index = code // 588
    medial_index = (code - initial_index * 588) // 28
    final_index = code % 28
    
    return [initial_list[initial_index], medial_list[medial_index], final_list[final_index]]

def decompose_korean_char(text):
    return [_decompose_korean_char(c) for c in text]
    
def _compose_korean_char(initial, medial, final=None, initial_p=None, medial_p=None, final_p=None):
    # try:
    initial_index = initial_list.index(initial) if (initial is not None) and (initial in initial_list) else None
    medial_index = medial_list.index(medial) if (medial is not None) and (medial in medial_list) else None
    final_index = final_list.index(final) if (final is not None) and (final in final_list) else None
    
    # 자소가 1개 이하인 경우 => 그나마 가장 괜찮을 걸로 대체
    if [initial_index, medial_index, final_index].count(None) >= 2:
        grapheme = [initial, medial, final]
        p = [initial_p, medial_p, final_p]
        idx = np.array(p).argmax()
        return [grapheme[idx], p[idx]]
    
    # 자소가 2개 이상인 경우 => 자소가 아닌 녀석만 기본 자소로 사용
    else:
        initial_index = initial_index if initial_index is not None else 11
        medial_index = medial_index if medial_index is not None else 0
        final_index = final_index if final_index is not None else 0
        
    char_code = 44032 + (initial_index * 21 + medial_index) * 28 + final_index
    return [chr(char_code), sum([initial_p, medial_p, final_p])/3]

def compose_korean_char(f, s, th, fp, sp, thp, first_main=False):
    if first_main:
        for i, x in enumerate(f):
            if x not in initial_list:
                s = s[:i]+" "+s[i:]
                th = th[:i]+" "+th[i:]
                sp.insert(i, 0)
                thp.insert(i, 0)
                
    composed = [_compose_korean_char(*c) for c in zip(f, s, th, fp, sp, thp)]
    char = "".join([x[0] for x in composed])
    conf = [x[1] for x in composed]
    return [char, conf]

