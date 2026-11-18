MED reqif_exporter.py:51,114,124,145 control chars unescaped -> _xml_safe() strip [\x00-\x08\x0b\x0c\x0e-\x1f￾￿]
MED sysml_xmi_exporter.py:263,273 same
MED reqif_exporter.py:166-186 cluster -1 dup ids -> skip -1/None
MED sysml_xmi_exporter.py:268-279 same -1 dup xmi:id
MED reqif:139,176 xmi:251 db_id None -> collisions; fallback index
MED providers.py:164-174 json strict -> strict=False
MED enrichment.py:535-543 wait_for budget < retry budget; single retry owner
MED enrichment.py:546-553 cache write failure marks success as error
MED providers.py:381-389,642 OSError/HTTPException escape -> ProviderResponseError; edge_explanation.py:241 500
MED providers.py:494-505 retries on 4xx; only retry 5xx/429/parse; drop response_format only after 400/parse
MED prompts.py:68-94,166-185 + refinement.py:611-656 prompt injection -> tag-delimit user data
MED pdf_report.py:60-75 latin1 font + control chars
MED refinement.py:592-600,649 sync per-cluster LLM, no breaker -> fallback after first failure
LOW jama_connector.py:320,328 int(getenv) at import
LOW jama_connector.py:341 'Noise' mislabel
LOW reqif_exporter.py:53 version 1.0 vs 1.2
LOW feedback_analyst.py:292 empty keyword match
LOW providers.py:556 Local no retries/fallback
LOW pdf_report.py:175 size None sort; pyplot thread safety
ARCH: 3 retry mechanisms; no prompt data boundary/no exporter validity tests; exporter normalisation duplicated x3
