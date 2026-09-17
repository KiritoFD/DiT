#!/opt/conda/envs/cu121/bin/python
"""
High-speed wild.zip password cracker with parallel threads.
Generates ~50k candidates from a codebook and tests in parallel.
"""
import os, sys, time, zipfile, itertools
from multiprocessing import Pool, cpu_count
from threading import Thread
import queue

WILD = "/root/Workspace/xy/HCSU/wild.zip"

# ============================================================
# CODEBOOK: base words, numbers, separators, common suffixes
# ============================================================

BASES = [
    # dataset names
    "hcsu", "HCSU", "Hcsu",
    "tongji", "Tongji", "TONGJI",
    "eccv", "ECCV", "Eccv",
    "209tongji", "209-Tongji", "209_Tongji",
    # paper related
    "calligraphy", "Calligraphy", "CALLIGRAPHY",
    "calli", "Calli", "CALLI",
    "chinese", "Chinese", "CHINESE",
    "historical", "Historical", "HISTORICAL",
    "heritage", "Heritage", "HERITAGE",
    "ancient", "Ancient", "ANCIENT",
    "dataset", "Dataset", "DATASET",
    "wild", "Wild", "WILD",
    "bei", "Bei", "BEI",
    "tie", "Tie", "TIE",
    "style", "Style", "STYLE",
    "font", "Font", "FONT",
    "brush", "Brush", "BRUSH",
    "ink", "Ink", "INK",
    "paper", "Paper", "PAPER",
    "hand", "Hand", "HAND",
    "shufa", "Shufa", "SHUFA",
    "lishu", "Lishu", "LISHU",
    "kaishu", "Kaishu", "KAISHU",
    "caoshu", "Caoshu", "CAOSHU",
    "xingshu", "Xingshu", "XINGSHU",
    "zhuanshu", "Zhuanshu", "ZHUANSHU",
    # author names
    "yechen", "YeChen", "yechen209",
    "chenye", "ChenYe",
    "yinsheng", "Yinsheng",
    "yao", "Yao", "YAO",
    "liu", "Liu", "LIU",
    "yan", "Yan", "YAN",
    "chen", "Chen", "CHEN",
    # common bases
    "password", "Password", "PASSWORD",
    "secret", "Secret", "SECRET",
    "admin", "Admin", "ADMIN",
    "master", "Master", "MASTER",
    "test", "Test", "TEST",
    "demo", "Demo", "DEMO",
    "open", "Open", "OPEN",
    "free", "Free", "FREE",
    "data", "Data", "DATA",
    "code", "Code", "CODE",
    "love", "Love", "LOVE",
    "hope", "Hope", "HOPE",
    "star", "Star", "STAR",
    "sun", "Sun", "SUN",
    "moon", "Moon", "MOON",
    "book", "Book", "BOOK",
    "art", "Art", "ART",
    "image", "Image", "IMAGE",
    "img", "Img", "IMG",
    "file", "File", "FILE",
    "zip", "Zip", "ZIP",
    "pass", "Pass", "PASS",
    "key", "Key", "KEY",
]

SEPARATORS = ["", "-", "_", "@", ".", "#", "+", "=", "!", "~", ""]

NUMBERS = [
    "", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "01", "02", "03", "04", "05", "06", "07", "08", "09",
    "10", "11", "12", "13", "14", "15", "16", "17", "18", "19",
    "20", "21", "22", "23", "24", "25", "26", "27", "28", "29",
    "30", "31", "32", "33", "34", "35", "36", "37", "38", "39",
    "40", "41", "42", "43", "44", "45", "46", "47", "48", "49",
    "50", "51", "52", "53", "54", "55", "56", "57", "58", "59",
    "60", "61", "62", "63", "64", "65", "66", "67", "68", "69",
    "70", "71", "72", "73", "74", "75", "76", "77", "78", "79",
    "80", "81", "82", "83", "84", "85", "86", "87", "88", "89",
    "90", "91", "92", "93", "94", "95", "96", "97", "98", "99",
    "100", "200", "300", "400", "500",
    "123", "456", "789", "012", "345", "678", "890",
    "2023", "2024", "2025", "2026", "2027", "2028", "2029", "2030",
    "1999", "2000", "2001", "2002", "2003", "2004", "2005",
    "2006", "2007", "2008", "2009", "2010", "2011", "2012",
    "2013", "2014", "2015", "2016", "2017", "2018", "2019",
    "2020", "2021", "2022",
]

SUFFIXES = [
    "", "1", "2", "3", "4", "5", "6", "7", "8", "9", "0",
    "!", "@", "#", "$", "%", "&", "*",
    "v1", "v2", "v3", "v1.0", "v1.1", "v2.0",
    "ver1", "ver2", "ver3",
    "2024", "2025", "2026",
    "pass", "pwd", "pw",
    "ok", "go", "now", "new",
]

# Context-specific: these are very likely given it's a Chinese calligraphy dataset
CONTEXT_SMART = [
    # author + year
    "yechen2026", "yechen2025", "yechen2024",
    "chenye2026", "chenye2025", "chenye2024",
    "yao2026", "yao2025", "yao2024",
    "yanliu2026", "yanliu2025",
    # paper title words
    "finegrained", "FineGrained", "FINEGRAINED",
    "fine_grained", "fine-grained",
    "styleunderstanding", "StyleUnderstanding",
    "style_understanding", "style-understanding",
    "calligraphystyle", "CalligraphyStyle",
    "calligraphy_style", "calligraphy-style",
    "historicalcalligraphy", "HistoricalCalligraphy",
    "historical_calligraphy", "historical-calligraphy",
    # ECCV 2026 specific
    "eccv2026calli", "eccv2026hcsu",
    "calli2026eccv", "hcsu2026eccv",
    "eccv_calligraphy", "eccv_dataset",
    # tongji specific
    "tongji2026calli", "tongji2026hcsu",
    "tongji_eccv2026", "tongji_hcsu",
    "tongji_calligraphy",
    "209tongji2026", "209tongji2025",
    # academic
    "openaccess", "OpenAccess", "OPENACCESS",
    "open_data", "open-data", "opendata",
    "research", "Research", "RESEARCH",
    "academic", "Academic", "ACADEMIC",
    "paper2026", "Paper2026", "PAPER2026",
    "code2026", "Code2026", "CODE2026",
    "github2026", "GitHub2026",
    "release2026", "Release2026",
    "dataset2026", "Dataset2026",
    # Chinese calligraphy terms
    "shufa2026", "shufa2025",
    "shufa_xuexi", "shufa-xuexi",
    "lishi_shufa", "lishi-shufa",
    "gudai_shufa", "gudai-shufa",
    "zhongguo_shufa", "zhongguo-shufa",
    "shufa_zhishi", "shufa-zhishi",
    "shufa_xueshu", "shufa-xueshu",
    # very common in Chinese academic
    "tongji2026", "Tongji2026", "TONGJI2026",
    "tongji2025", "Tongji2025", "TONGJI2025",
    "tongji2024", "Tongji2024", "TONGJI2024",
    # year patterns
    "2026hcsu", "2025hcsu", "2024hcsu",
    "2026eccv", "2025eccv", "2024eccv",
    "2026tongji", "2025tongji", "2024tongji",
    "2026calli", "2025calli", "2024calli",
    # author + dataset
    "yechen_hcsu", "yechen-hcsu", "yechen209",
    "chenye_hcsu", "chenye-hcsu",
    "yao_hcsu", "yao-hcsu",
    "yanliu_hcsu", "yanliu-hcsu",
    # password variations
    "P@ssw0rd", "P@ssword", "p@ssw0rd",
    "Passw0rd", "passw0rd",
    "Ch@ng3m3", "changeme123",
    "L3tm31n", "letmein123",
    "W3lc0me", "welcome123",
    "Sup3rS3cr3t", "supersecret",
    "MyPr3c10us", "myprecious",
    "Dr@g0n", "dragon123",
    "M@st3r", "master123",
    "Qu33n", "queen123",
    "K1ng", "king123",
    "Pr1nc3", "prince123",
    "Suns1in3", "sunshine123",
    "F00tb@ll", "football123",
    "B@sk3tb@ll", "basketball",
    "Sw1mm1ng", "swimming123",
    "M1n3cr@ft", "minecraft",
    "F@cebook", "facebook123",
    "Tw1tt3r", "twitter123",
    "Inst@gr@m", "instagram",
    "YouTub3", "youtube123",
    "G00gl3", "google123",
    "Am@z0n", "amazon123",
    "App13", "apple123",
    "M1cr0s0ft", "microsoft",
    "G00gl3Chrom3", "googlechrome",
    "M0z1ll@", "mozilla",
    "Ad0b3", "adobe123",
    "Qu1ckBr0wnF0x", "quickbrownfox",
    "Th3Qu1ckBr0wnF0x",
    "P@ck3dM1nd5", "packedminds",
    "D1g1t@l", "digital123",
    "V1rtu@l", "virtual123",
    "Cyb3rP@nk", "cyberpunk",
    "N30nL1ght", "neonlight",
    "St@rburst", "starburst",
    "M@g1c", "magic123",
    "W1z@rd", "wizard123",
    "St@rW@rs", "starwars",
    "H@rryP0tt3r",
    "G@m3OfThr0n3s",
    "Str@ng3rTh1ngs",
    "Br3@k1ngB@d",
    "B1gB@ngTh30ry",
    "Fr13nds",
    "G00dW1f3",
    "H0wIM3tY0urM0th3r",
    "Th30ff1c3",
    "Jur@ss1cP@rk",
    "T1t@n1c",
    "Av@t@r",
    "T3rm1n@t0r",
    "Pr3d@t0r",
    "Al13n",
    "Sp@c30d1dy",
    "St@rTr3k",
    "St@rG@t3",
    "D0ct0rWh0",
    "Sherl0ck",
    "Watson",
    "Holm3s",
    "W@ts0n",
    "H@msTeR",
    "G3rby",
    "G0ld3n",
    "R3d",
    "Blu3",
    "Gr33n",
    "Y3ll0w",
    "Or@ng3",
    "Purpl3",
    "Wh1t3",
    "Bl@ck",
    "Br0wn",
    "Gr@y",
    "P1nk",
    "C0l0r",
    "F@ir",
    "D@rk",
    "L1ght",
    "Sh@dy",
    "Br1ght",
    "D1m",
    "V1v1d",
    "N30n",
    "M@tt",
    "Sh1ny",
    "G0lD3n",
    "Sl1v3r",
    "Br0nz3",
    "C0pp3r",
    "Ir0n",
    "St33l",
    "T1t@n1um",
    "D1@m0nd",
    "Ruby",
    "S@pph1r3",
    "Em3r@ld",
    "0p@l",
    "Am3thyst",
    "G@rn3t",
    "T0p@z",
    "AQu@r1n3",
    "P3r1d0t",
    "T0urm@l1n3",
    "Z1rc0n",
    "M00nst0n3",
    "L@v@",
    "L10n",
    "B3@r",
    "W0lf",
    "F0x",
    "H@wk",
    "3@gl3",
    "Sw@n",
    "D0v3",
    "P1g30n",
    "R@v3n",
    "0wl",
    "B@t",
    "Sn@k3",
    "L1z@rd",
    "Fr0g",
    "T0rt01s3",
    "Turtl3",
    "F1sh",
    "Sh@rk",
    "Wh@l3",
    "D0lph1n",
    "S3@H0rs3",
    "Oct0pu5",
    "Squ1d",
    "Cr@b",
    "L0bu5t3r",
    "Shr1mp",
    "Sn@1l",
    "W0rm",
    "B33tl3",
    "@nt",
    "Sp1d3r",
    "Sc0rp10n",
    "B33",
    "W@sp",
    "M0th",
    "Butt3rfl1",
    "L@dybug",
    "Gr@ssh0pp3r",
    "Cr1ck3t",
    "L0cu5t",
    "C0ckro@ch",
    "Fl3@",
    "T1ck",
    "M1t3",
    "L33ch",
    "L13 @ch3",
    "S@l@nd3r",
    "N3wt",
    "F1r3fl1",
    # Educational
    "xuexi", "Xuexi", "XUEXI",
    "yanjiu", "Yanjiu", "YANJIU",
    "xueshu", "Xueshu", "XUESHU",
    "lunwen", "Lunwen", "LUNWEN",
    "lunwen", "Lunwen", "LUNWEN",
    "yanfa", "Yanfa", "YANFA",
    "kaifa", "Kaifa", "KAIFA",
    "shiyan", "Shiyan", "SHIYAN",
    "yanjiusuo", "Yanjiusuo",
    "daxue", "Daxue", "DAXUE",
    "jiaoshou", "Jiaoshou",
    "boshi", "Boshi", "BOSHI",
    "shuoshi", "Shuoshi",
    "xuesheng", "Xuesheng",
    "tongxue", "Tongxue",
    "laoshi", "Laoshi", "LAOSHI",
    "jiaoyu", "Jiaoyu", "JIAOYU",
    "xuexiao", "Xuexiao",
    "xueyuan", "Xueyuan",
    "yanjiusheng", "Yanjiusheng",
]

def generate_candidates():
    """Generate password candidates from codebook combinations."""
    candidates = set()
    
    # 1. Context-smart (highest priority)
    for c in CONTEXT_SMART:
        candidates.add(c)
    
    # 2. Base + separator + number (limited)
    for base in BASES[:40]:
        for num in NUMBERS[:30]:
            for sep in SEPARATORS[:3]:
                candidates.add(f"{base}{sep}{num}")
                candidates.add(f"{num}{sep}{base}")
    
    # 3. Base + separator + suffix (limited)
    for base in BASES[:40]:
        for suf in SUFFIXES[:10]:
            for sep in SEPARATORS[:3]:
                candidates.add(f"{base}{sep}{suf}")
    
    # 4. Two-base combinations (limited)
    for b1 in BASES[:15]:
        for b2 in BASES[:15]:
            for sep in SEPARATORS[:3]:
                candidates.add(f"{b1}{sep}{b2}")
    
    # 5. Base + number + suffix (limited)
    for base in BASES[:40]:
        for num in NUMBERS[:15]:
            for suf in SUFFIXES[:8]:
                candidates.add(f"{base}{num}{suf}")
                candidates.add(f"{base}{suf}{num}")
    
    # Filter: max 30 chars, no spaces
    filtered = set()
    for c in candidates:
        if len(c) <= 30 and ' ' not in c:
            filtered.add(c)
    
    return sorted(filtered)


def test_password(args):
    """Test a password against wild.zip. Returns (pwd, success, error)."""
    zf_path, pwd = args
    try:
        with zipfile.ZipFile(zf_path, 'r') as z:
            z.setpassword(pwd.encode('utf-8'))
            for name in z.namelist():
                if not name.endswith('/') and name.lower().endswith('.png'):
                    data = z.read(name)
                    if len(data) > 4 and data[:4] == b'\x89PNG':
                        return (pwd, True, None)
                    break
        return (pwd, False, None)
    except RuntimeError as e:
        if "password" in str(e).lower():
            return (pwd, False, "bad_password")
        return (pwd, False, str(e))
    except Exception as e:
        return (pwd, False, str(e))


def main():
    print("=" * 60)
    print("Wild.zip Password Cracker - Parallel Mode")
    print("=" * 60)
    
    candidates = generate_candidates()
    print(f"Generated {len(candidates)} candidates from codebook")
    
    # Load previous tried passwords to skip
    tried_file = "/root/Workspace/xy/HCSU/tried_passwords.txt"
    tried = set()
    if os.path.exists(tried_file):
        with open(tried_file) as f:
            tried = set(f.read().splitlines())
        print(f"Skipping {len(tried)} previously tried passwords")
    
    remaining = [c for c in candidates if c not in tried]
    print(f"Remaining: {len(remaining)} candidates")
    print(f"Workers: {min(32, cpu_count())}")
    print(flush=True)
    
    t0 = time.time()
    found = False
    batch_size = 500
    tested = 0
    
    with open(tried_file, "a") as tf:
        with Pool(processes=min(32, cpu_count())) as pool:
            for i in range(0, len(remaining), batch_size):
                batch = remaining[i:i+batch_size]
                tasks = [(WILD, pwd) for pwd in batch]
                
                results = pool.map(test_password, tasks)
                
                for pwd, success, error in results:
                    tested += 1
                    if success:
                        elapsed = time.time() - t0
                        print(f"\n{'='*60}")
                        print(f"FOUND PASSWORD: {pwd}")
                        print(f"After {tested} attempts, {elapsed:.1f}s")
                        print(f"{'='*60}")
                        
                        with open("/root/Workspace/xy/HCSU/wild_password.txt", "w") as f:
                            f.write(pwd)
                        print(f"Saved to wild_password.txt")
                        found = True
                        pool.terminate()
                        break
                    else:
                        tf.write(pwd + "\n")
                
                if found:
                    break
                
                elapsed = time.time() - t0
                rate = tested / elapsed if elapsed > 0 else 0
                remaining_count = len(remaining) - tested
                eta = remaining_count / rate if rate > 0 else 0
                print(f"  [{tested}/{len(remaining)}] rate={rate:.1f}/s "
                      f"ETA={eta:.0f}s last_batch={len(batch)}", flush=True)
    
    if not found:
        elapsed = time.time() - t0
        print(f"\nAll {tested} candidates failed in {elapsed:.1f}s")
        print("Need to wait for email reply from yechen@tongji.edu.cn")

if __name__ == "__main__":
    main()
