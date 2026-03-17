import urllib.request, json, time, concurrent.futures

URL = "http://localhost:8000/v1/chat/completions"

def req(prompt, mt=512):
    p = json.dumps({"model":"qwen3.5-35b","messages":[{"role":"user","content":prompt}],"max_tokens":mt,"temperature":0.7}).encode()
    r = urllib.request.Request(URL, data=p, headers={"Content-Type":"application/json"})
    s = time.time()
    d = json.loads(urllib.request.urlopen(r, timeout=180).read())
    e = time.time() - s
    u = d.get("usage",{})
    return {"ct":u.get("completion_tokens",0),"t":e}

print("=== SINGLE REQUEST ===")
r = req("Explain general relativity covering spacetime curvature and field equations.", 512)
print(f"Tokens: {r['ct']}, Time: {r['t']:.1f}s, Speed: {r['ct']/r['t']:.1f} tok/s")

ps = ["Explain quantum entanglement.","History of Rome.","How photosynthesis works.","Neural network backpropagation.","Water cycle and climate.","Thermodynamics.","Immune system.","Blockchain.","Nuclear fusion.","Vaccines."]
for n in [2, 5, 10]:
    print(f"\n=== {n} CONCURRENT ===")
    s = time.time()
    with concurrent.futures.ThreadPoolExecutor(n) as ex:
        rs = list(ex.map(lambda p: req(p, 256), ps[:n]))
    w = time.time() - s
    tc = sum(x["ct"] for x in rs)
    pr = [x["ct"]/x["t"] for x in rs]
    print(f"Total: {tc} tok, Wall: {w:.1f}s, Agg: {tc/w:.1f} tok/s, Per-req avg: {sum(pr)/len(pr):.1f} tok/s")
