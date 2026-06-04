# Phase 5: MBSE Export

Exports the clustered requirements, their groupings, and dependency links to
standard requirements-engineering and model-based formats.

| Format | `fmt` | Exporter | Imports into |
|--------|-------|----------|--------------|
| OMG ReqIF 1.2 | `reqif` | `export/reqif_exporter.py` | IBM DOORS Next, ReqView, Polarion, Eclipse RMF |
| SysML / UML XMI 2.5.1 | `sysml` | `export/sysml_xmi_exporter.py` | Eclipse Papyrus, MagicDraw / Cameo |
| Jama Connect | `jama` | `export/jama_connector.py` | Jama (REST upload or offline JSON import) |
| CSV | `csv` | `services/export_service.py` | Spreadsheets |

## API

```
GET /api/export/{reqif|sysml|jama|csv}?session_id=<id>
```

Returns the file with the correct media type and a `Content-Disposition`
attachment header. The **Export** page provides download cards.

## Mapping

- **ReqIF**: requirements become `SPEC-OBJECT`s (ReqID + XHTML Text + Module +
  Cluster); clusters become heading `SPEC-OBJECT`s grouping their requirements in
  a `SPEC-HIERARCHY`. Noise requirements sit at the specification root.
- **SysML XMI**: the session is a `uml:Model`; each cluster is a `uml:Package`;
  each requirement is a `uml:Class` keyworded `«requirement»` with its text as an
  owned comment; dependency-tree edges become `uml:Dependency` elements keyworded
  `«deriveReqt»` / `«trace»`.
- **Jama**: an item bundle (`POST /rest/v1/items` shape) plus relationships from
  the dependency tree. Configure the target project via environment:
  `JAMA_BASE_URL`, `JAMA_API_TOKEN`, `JAMA_PROJECT_ID`, `JAMA_ITEM_TYPE_ID`,
  `JAMA_RELATIONSHIP_TYPE_ID`.

## Dependency links

ReqIF/CSV carry cluster grouping; SysML and Jama additionally include dependency
relationships **when a dependency tree has been generated** for the session
(`POST /api/dependencies/generate`).

## Fidelity note

ReqIF output is schema-shaped and tool-importable. SysML/UML profile stereotype
application is tool-specific; the exporter emits portable standards XMI with
SysML keywords, which Papyrus and MagicDraw import as a model tree. Validate
against a target tool before production use.
