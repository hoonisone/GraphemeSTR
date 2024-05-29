from PIL import Image
import numpy as np

def rotate_image(image, angle):
    # 이미지를 원하는 각도로 회전
    rotated_image = image.rotate(angle, expand=True)
    
    # 회전 후 이미지의 크기 계산
    width, height = rotated_image.size
    
    # 새로운 이미지를 생성하여 회전된 이미지를 중앙에 배치
    new_image = Image.new("RGB", (width, height), (255, 255, 255))
    x_offset = (width - image.width) // 2
    y_offset = (height - image.height) // 2
    
    # new_image.paste(rotated_image, (x_offset, y_offset))
    new_image.paste(rotated_image, (0, 0))
    
    return new_image
class RotateVertical(object):
    def __init__(self, **kwargs):
        self.aspect_ratio_thresh = kwargs["aspect_ratio_thresh"]
        self.angle = kwargs["angle"]

    def __call__(self, data):
        img = data['image']
        h, w, c = img.shape
        if h/w > self.aspect_ratio_thresh:
            img = rotate_image(Image.fromarray(img), self.angle)
            img = np.array(img)
            data["image"] = img
            
        return data

jamo_to_choseong = {
            'ᄀ': '가', 'ᄁ': '까', 'ᄂ': '나', 'ᄃ': '다', 'ᄄ': '따', 'ᄅ': '라',
            'ᄆ': '마', 'ᄇ': '바', 'ᄈ': '빠', 'ᄉ': '사', 'ᄊ': '싸', 'ᄋ': '아',
            'ᄌ': '자', 'ᄍ': '짜', 'ᄎ': '차', 'ᄏ': '카', 'ᄐ': '타', 'ᄑ': '파', 'ᄒ': '하'
        }

def extract_first_grapheme(text):
    result = ''
    for char in text:
        # 한글 유니코드 범위인지 확인
        if 44032 <= ord(char) <= 55203:
            # 초성 구하기
            choseong_index = (ord(char) - 44032) // 588
            # 초성을 문자로 변환하여 결과에 추가
            choseong = chr(choseong_index + 0x1100)  # 초성 유니코드 시작값은 0x1100입니다.
            result += jamo_to_choseong[choseong]
        else: # 한글이 아니면 그대로 결과에 추가
            result += char
            
    return result


# # 예시 사용
# text = "안녕하세요, Hello!"
# result = convert_to_choseong(text)
# print(result)
  
class GetFirstGrapheme(object):
    def __init__(self, **kwargs):
        pass

    def __call__(self, data):       
        # print(data["label"])
        data["label"] = convert_to_choseong(data["label"])
        # print(data["label"])     
        return data
    
    
import functools


first_grapheme_dict = {
            'ᄀ': '가', 'ᄁ': '까', 'ᄂ': '나', 'ᄃ': '다', 'ᄄ': '따', 'ᄅ': '라',
            'ᄆ': '마', 'ᄇ': '바', 'ᄈ': '빠', 'ᄉ': '사', 'ᄊ': '싸', 'ᄋ': '아',
            'ᄌ': '자', 'ᄍ': '짜', 'ᄎ': '차', 'ᄏ': '카', 'ᄐ': '타', 'ᄑ': '파', 'ᄒ': '하'
        }


def extract_first_grapheme(text, type="first"):
    result = ''
    for char in text:
        # 한글 유니코드 범위인지 확인
        if 44032 <= ord(char) <= 55203:
            # 초성 구하기
            choseong_index = (ord(char) - 44032) // 588
            # 초성을 문자로 변환하여 결과에 추가
            choseong = chr(choseong_index + 0x1100)  # 초성 유니코드 시작값은 0x1100입니다.
            result += first_grapheme_dict[choseong]
        else: # 한글이 아니면 그대로 결과에 추가
            result += char
            
    return result

sceond_grapheme_dict = {"ᅡ":"아","ᅢ":"애","ᅣ":"야","ᅤ":"얘","ᅥ":"어","ᅦ":"에","ᅧ":"여",
"ᅨ":"예","ᅩ":"오","ᅪ":"와","ᅫ":"왜","ᅬ":"외","ᅭ":"요","ᅮ":"우",
"ᅯ":"워","ᅰ":"웨","ᅱ":"위","ᅲ":"유","ᅳ":"으","ᅴ":"의","ᅵ":"이"}


def extract_second_grapheme(text):
    result = ''
    for char in text:
        # 한글 유니코드 범위인지 확인
        if 44032 <= ord(char) <= 55203:
            # 중성 구하기
            jungseong_index = ((ord(char) - 44032) % 588) // 28
            # 중성을 문자로 변환하여 결과에 추가
            jungseong = chr(jungseong_index + 0x1161)  # 중성 유니코드 시작값은 0x1161입니다.
            result += sceond_grapheme_dict[jungseong]
        else:
            # 한글이 아니면 그대로 결과에 추가
            result += char
        
    return result

third_grapheme_dict = {"ᆨ":"윽","ᆫ":"은","ᆮ":"읃","ᆯ":"을","ᆷ":"음","ᆸ":"읍","ᆺ":"읏",
"ᆼ":"응","ᆽ":"읒","ᆾ":"읓","ᆿ":"읔","ᇀ":"읕","ᇁ":"읖","ᇂ":"읗",
"ᆩ":"윾","ᆻ":"읐","ᆪ":"윿","ᆬ":"읁","ᆭ":"읂","ᆰ":"읅","ᆱ":"읆",
"ᆲ":"읇","ᆳ":"읈","ᆴ":"읉","ᆵ":"읊","ᆶ":"읋","ᆹ":"읎"}


def extract_third_grapheme(text):
    result = ''
    has_jongseong = False  # 받침이 있는지 여부를 판별하기 위한 변수

    for char in text:
        # 한글 유니코드 범위인지 확인
        if 44032 <= ord(char) <= 55203:
            # 종성 구하기
            jongseong_index = (ord(char) - 44032) % 28

            # 받침이 있는 경우에만 결과에 추가하고, 받침이 없는 경우 "으"를 추가
            if jongseong_index != 0:
                jongseong = chr(jongseong_index + 0x11A7)  # 종성 유니코드 시작값은 0x11A8이 아닌 0x11A7입니다.
                result += third_grapheme_dict[jongseong]
            else:
                result += '으'
        else:
            # 한글이 아니면 그대로 결과에 추가
            result += char
    
    return result

# @functools.lru_cache(maxsize=1000000)
def extract_grapheme(text):
    return {
        "first":extract_first_grapheme(text),
        "second":extract_second_grapheme(text),
        "third":extract_third_grapheme(text),
        "origin":text
    }
    
class ExtractGrapheme(object):
    def __init__(self, **kwargs):
        pass

    def __call__(self, data):       
        data["label"] = extract_grapheme(data["label"])
        return data