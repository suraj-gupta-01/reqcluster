HIGH refinement_service.py:217 delete before N LLM calls -> SQLite write lock held; compute then short txn
MED feedback_service.py:71 is_noise/membership_prob not updated on move [confirmed]
MED feedback_service.py:41 None vs -1 noise; reject size decrement out of order -> normalize, recount
MED routes.py:242 re-cluster leaves stale Suggestion/Feedback/Constraint/Iteration/DepTree/Explanation rows
MED active_learning_service.py:140 recluster doesn't invalidate suggestions; apply merge/split no existence check; positional sub_labels
MED dependency_service.py:150 stale DependencyExplanation cache after regen
MED routes.py:88 upload: bad xlsx->500, .xls openpyxl, case-sensitive ext, full read before size cap, sync preprocess on loop, per-row add
MED enrichment_service.py:229 failure upsert destroys good row on force_refresh
MED enrichment_service.py:388 N+1, O(n^2) errors scan, sync DB in async
MED routes.py:584 bogus provider_name -> 500 (str not Literal) schemas.py:166,184,237 [confirmed]
MED routes.py:668 feedback not-found -> 400 not 404 [confirmed]
LOW routes.py:421 list endpoints don't 404 unknown session
LOW routes.py:139 /enrich shares pipeline_progress key; concurrent enrich unique-violation 500
LOW feedback_service.py:223 CSV formula injection -> _csv_safe
LOW routes.py:546 sort_field silently ignored; page w/o page_size; LIKE wildcards unescaped; dup expose-headers
LOW main.py:66 no auth/rate limit (IDOR) -> auth dep or localhost bind
LOW refinement_service.py:335 duplicate summary LLM calls; coherence 0.0 treated missing
LOW routes.py:375 progress_cache stale; dead: SessionCreate, ProgressUpdate, validate_session_enrichment_ready
ARCH: cluster state across ~10 tables, no invalidation rule/lock; sync DB in async + LLM holding txn; inconsistent errors, no auth
