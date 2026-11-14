import subprocess, time, glob, threading, urllib.request, json, sys

MODE = sys.argv[1]
MULT = int(sys.argv[2]) if len(sys.argv) > 2 else 800


def ranks():
    out = subprocess.run(['ps', '-eo', 'pid,args'], capture_output=True, text=True).stdout
    d = {}
    for line in out.splitlines():
        if 'Worker_PP' in line:
            parts = line.split(None, 1)
            try:
                pid = int(parts[0])
            except Exception:
                continue
            tail = parts[1].split('Worker_PP')[-1].strip()
            if tail.isdigit():
                d[int(tail)] = pid
    return d


def ticks(pid):
    s = open('/proc/%d/stat' % pid).read().split()
    return int(s[13]) + int(s[14])


def freq():
    v = []
    for p in glob.glob('/sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq'):
        try:
            v.append(int(open(p).read()) / 1000.0)
        except Exception:
            pass
    return sum(v) / len(v) if v else -1


para = '在分布式推理系统中，流水线并行把模型按层切分到多张卡上，每一段负责一部分层，段与段之间通过点对点通信传递激活值。'
if MODE == 'prefill':
    body = {'model': 'deepseek-v4.1-flash-1017',
            'messages': [{'role': 'user', 'content': para * MULT}],
            'max_tokens': 8, 'stream': False}
    dur = 40
else:
    body = {'model': 'deepseek-v4.1-flash-1017',
            'messages': [{'role': 'user', 'content': '请详细讲解流水线并行的原理'}],
            'max_tokens': 500, 'stream': False}
    dur = 14

result = {}


def send():
    req = urllib.request.Request('http://127.0.0.1:9010/v1/chat/completions',
                                 data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    try:
        r = json.loads(urllib.request.urlopen(req, timeout=600).read())
        result['usage'] = r.get('usage')
    except Exception as e:
        print('req err', e)


time.sleep(1)
p = ranks()
base = {r: ticks(pid) for r, pid in p.items()}
t0 = time.time()
th = threading.Thread(target=send)
th.start()
print('t  freqMHz  ' + '  '.join('PP%d' % r for r in sorted(p)) + '  total')
while time.time() - t0 < dur:
    time.sleep(1.0)
    row = []
    for r in sorted(p):
        row.append((ticks(p[r]) - base[r]) / 100.0)
    base = {r: ticks(pid) for r, pid in p.items()}
    print('%4.1f  %6.1f  ' % (time.time() - t0, freq()) + '  '.join('%5.2f' % x for x in row) + '  %5.2f' % sum(row))
    if 'usage' in result and time.time() - t0 > 3:
        break
th.join()
print('usage:', result.get('usage'))
print('governor:', open('/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor').read().strip(),
      'max:', int(open('/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq').read()) / 1000.0)
