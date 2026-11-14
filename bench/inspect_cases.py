import json

d = json.load(open("cases_v1.json"))["cases"]
for k, v in d.items():
    p = v.get("prompts") or []
    print(k, "kind=", v.get("kind"), "max_tokens=", v.get("max_tokens"),
          "conc=", v.get("conc"), "prompt_chars=", len(p[0]) if p else 0)
