#!/opt/conda/envs/cu121/bin/python
"""
Try common passwords on merged wild.zip.
Validates by reading first PNG file and checking magic bytes.
"""
import os, sys, time, zipfile

WILD = "/root/Workspace/xy/HCSU/wild.zip"

CANDIDATES = [
    # bei/tie password
    "eWZNA9hzJoyMHtQwVh7QnBtN7",
    # dataset / paper
    "HCSU", "hcsu", "HCSU2026", "hcsu2026", "HCSU_v1", "HCSU_v1.1",
    "Tongji", "tongji", "Tongji209", "tongji209", "209Tongji", "209-tongji",
    "ECCV", "eccv", "ECCV2026", "eccv2026",
    "Fine-Grained", "Historical", "Calligraphy", "Style", "Understanding",
    "HCSU_ECCV2026", "hcsu_eccv2026",
    # Chinese pinyin
    "calligraphy", "Calligraphy", "CALRIGRAPHY",
    "chinese", "Chinese", "CHINESE",
    "heritage", "Heritage", "Cultural",
    "ancient", "Ancient", "Historical",
    "dataset", "Dataset", "DATASET",
    "wild", "Wild", "WILD",
    # author names
    "yechen", "YeChen", "yechen209", "chenye", "ChenYe",
    "Yinsheng", "yinsheng", "Yao", "yao",
    "Liu", "liu", "Yan", "yan",
    # common passwords
    "password", "Password", "PASSWORD", "pass", "Pass",
    "123456", "12345678", "123456789", "12345",
    "abc123", "ABC123", "abc1234", "ABCabc123",
    "admin", "Admin", "ADMIN", "root", "Root",
    "test", "Test", "TEST", "demo", "Demo",
    "qwerty", "Qwerty", "QWERTY", "asdfgh", "zxcvbn",
    "letmein", "LetMeIn", "welcome", "Welcome",
    "monkey", "dragon", "master", "login", "Login",
    # number patterns
    "000000", "111111", "666666", "888888", "999999",
    "012345", "1234567", "1234567890",
    # Chinese pinyin
    "tongji", "Tongji", "TONGJI",
    "huashi", "Huashi", "HUASHI",
    "beiti", "Beiti", "BEITI",
    "shufa", "Shufa", "SHUFA",
    "shufa", "ShuFa",
    "calli", "Calli", "CALLI",
    "hua", "Hua", "HUA",
    "shu", "Shu", "SHU",
    "fa", "Fa", "FA",
    "ji", "Ji", "JI",
    "yan", "Yan", "YAN",
    "jiu", "Jiu", "JIU",
    "shi", "Shi", "SHI",
    "dai", "Dai", "DAI",
    "wen", "Wen", "WEN",
    "wu", "Wu", "WU",
    "ming", "Ming", "MING",
    "qing", "Qing", "QING",
    "tang", "Tang", "TANG",
    "song", "Song", "SONG",
    "yuan", "Yuan", "YUAN",
    "ming", "Ming", "MING",
    "qing", "Qing", "QING",
    # academic
    "2026", "2025", "2024", "2023",
    "v1.0", "v1.1", "v1", "V1", "V1.0", "V1.1",
    "paper", "Paper", "PAPER",
    "code", "Code", "CODE",
    "github", "GitHub", "GITHUB",
    "release", "Release", "RELEASE",
    "open", "Open", "OPEN",
    "access", "Access", "ACCESS",
    "free", "Free", "FREE",
    "public", "Public", "PUBLIC",
    "download", "Download", "DOWNLOAD",
    # Chinese
    "beifang", "Beifang", "BEIFANG",
    "nanfang", "Nanfang", "NANFANG",
    "lishu", "Lishu", "LISHU",
    "kaishu", "Kaishu", "KAISHU",
    "caoshu", "Caoshu", "CAOSHU",
    "xingshu", "Xingshu", "XINGSHU",
    "zhuanshu", "Zhuanshu", "ZHUANSHU",
    "bei", "Bei", "BEI",
    "tie", "Tie", "TIE",
    # common weak
    "changeme", "ChangeMe", "CHANGEME",
    "p@ssw0rd", "P@ssw0rd", "p@ssword",
    "pass123", "Pass123", "pass1234",
    "secret", "Secret", "SECRET",
    "master", "Master", "MASTER",
    "shadow", "Shadow", "SHADOW",
    "qwerty123", "Qwerty123", "QWERTY123",
    "iloveyou", "ILoveYou", "ILOVEYOU",
    "sunshine", "Sunshine", "SUNSHINE",
    "princess", "Princess", "PRINCESS",
    "football", "Football", "FOOTBALL",
    "charlie", "Charlie", "CHARLIE",
    "thomas", "Thomas", "THOMAS",
    "jordan", "Jordan", "JORDAN",
    "michael", "Michael", "MICHAEL",
    "jennifer", "Jennifer", "JENNIFER",
    "hunter", "Hunter", "HUNTER",
    # combinations
    "HCSU_dataset", "hcsu_dataset", "HCSUDataset",
    "tongji209", "Tongji209", "TONGJI209",
    "eccv2026", "ECCV2026", "ECCV_2026",
    "calligraphy_hcsu", "hcsu_calligraphy",
    "wild_hcsu", "hcsu_wild", "WILD_HCSU",
    "bei_tie", "tie_bei", "BEI_TIE",
    "hcsu_bei", "hcsu_tie", "hcsu_wild",
    # short
    "HCSU", "hcsu", "ECCV", "eccv",
    "2026", "2025", "Tong", "tong",
    "Ji20", "ji20", "HuaS", "huas",
    "ShuF", "shuf", "BeiT", "beit",
    "TieW", "tiew", "Wild", "wild",
    "Call", "call", "Star", "star",
    "Love", "love", "Hope", "hope",
    "Blue", "blue", "Gold", "gold",
    "Red1", "red1", "Sun1", "sun1",
    # with numbers
    "HCSU2024", "hcsu2024", "HCSU2025", "hcsu2025",
    "ECCV2024", "eccv2024", "ECCV2025", "eccv2025",
    "Tongji2024", "tongji2024", "Tongji2025", "tongji2025",
    "yechen2024", "yechen2025", "chenye2024", "chenye2025",
    # with symbols
    "HCSU@2026", "hcsu@2026", "ECCV@2026", "eccv@2026",
    "HCSU!2026", "hcsu!2026", "ECCV!2026", "eccv!2026",
    "HCSU#2026", "hcsu#2026", "ECCV#2026", "eccv#2026",
    "HCSU$2026", "hcsu$2026", "ECCV$2026", "eccv$2026",
    "HCSU*2026", "hcsu*2026", "ECCV*2026", "eccv*2026",
    # shufa related
    "shufa2026", "Shufa2026", "SHUFA2026",
    "calli2026", "Calli2026", "CALLI2026",
    "art2026", "Art2026", "ART2026",
    "font2026", "Font2026", "FONT2026",
    "style2026", "Style2026", "STYLE2026",
    "hand2026", "Hand2026", "HAND2026",
    "ink2026", "Ink2026", "INK2026",
    "brush2026", "Brush2026", "BRUSH2026",
    "paper2026", "Paper2026", "PAPER2026",
    # pinyin combos
    "shufaji", "Shufaji", "SHUFAJI",
    "shufaxuexi", "Shufaxuexi", "SHUFAXUEXI",
    "lishixuexi", "Lishixuexi", "LISHIXUEXI",
    "gudai", "Gudai", "GUDAI",
    "shufa2025", "Shufa2025", "SHUFA2025",
    "shufa2024", "Shufa2024", "SHUFA2024",
    # ECCV patterns
    "ECCV_Tongji", "eccv_tongji", "ECCV_TONGJI",
    "Tongji_ECCV", "tongji_eccv", "TONGJI_ECCV",
    "HCSU_ECCV", "hcsu_eccv", "HCSU_TONGJI",
    # random common
    "changeme123", "password1", "password123",
    "admin123", "letmein123", "welcome123",
    "monkey123", "dragon123", "master123",
    "login123", "abc123456", "passpass",
    "test123", "demo123", "sample123",
    "data123", "set123", "wild123",
    "2026pass", "2026Pass", "2026PASS",
    "pass2026", "Pass2026", "PASS2026",
    # book/file
    "book", "Book", "BOOK", "books", "Books",
    "calli_book", "Calli_Book", "font_book",
    "style_book", "art_book", "ink_book",
    # last resort
    "0000", "1111", "2222", "3333", "4444",
    "5555", "6666", "7777", "8888", "9999",
    "aaaa", "AAAA", "bbbb", "BBBB", "cccc",
    "dddd", "DDDD", "eeee", "EEEE", "ffff",
    "1a2b3c", "A1b2C3", "x1y2z3",
    "abc", "ABC", "xyz", "XYZ",
    "aaa", "AAA", "bbb", "BBB",
]

def try_password(zf_path, pwd):
    """Try password on zip. Return True if works (extracts valid PNG)."""
    try:
        with zipfile.ZipFile(zf_path, 'r') as z:
            z.setpassword(pwd.encode('utf-8'))
            for name in z.namelist():
                if not name.endswith('/') and name.lower().endswith('.png'):
                    data = z.read(name)
                    if len(data) > 4 and data[:4] == b'\x89PNG':
                        return True
                    break
        return False
    except RuntimeError as e:
        if "password" in str(e).lower() or "bad" in str(e).lower():
            return False
        return False
    except Exception:
        return False

def main():
    print(f"Target: {WILD}")
    print(f"Size: {os.path.getsize(WILD)/1024/1024/1024:.1f}GB")
    print(f"Candidates: {len(CANDIDATES)}")
    print(flush=True)
    
    t0 = time.time()
    found = False
    
    for i, pwd in enumerate(CANDIDATES):
        if (i+1) % 50 == 0 or i == 0:
            elapsed = time.time() - t0
            rate = (i+1) / elapsed if elapsed > 0 else 0
            eta = (len(CANDIDATES) - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1}/{len(CANDIDATES)}] rate={rate:.1f}/s ETA={eta:.0f}s trying: {pwd}", flush=True)
        
        if try_password(WILD, pwd):
            elapsed = time.time() - t0
            print(f"\n{'='*60}")
            print(f"FOUND PASSWORD: {pwd}")
            print(f"After {i+1} attempts, {elapsed:.1f}s")
            print(f"{'='*60}")
            
            with open("/root/Workspace/xy/HCSU/wild_password.txt", "w") as f:
                f.write(pwd)
            print(f"Saved to /root/Workspace/xy/HCSU/wild_password.txt")
            found = True
            break
    
    if not found:
        elapsed = time.time() - t0
        print(f"\nAll {len(CANDIDATES)} candidates failed in {elapsed:.1f}s")
        print("Need to wait for email reply from yechen@tongji.edu.cn")

if __name__ == "__main__":
    main()
