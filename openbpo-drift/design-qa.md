# Design QA

Target: user-supplied multi-tab OpenBPO Drift desktop mockups dated June 3, 2026.

Compared areas:
- sidebar shell
- page header and local-processing banner
- Data Preview layout
- Schema Mapper layout
- Data Quality layout
- Drift Alerts layout
- KPI Explorer layout
- Export layout

Fixed during this pass:
- upgraded the sidebar and header shell to match the mockup hierarchy more closely
- rebuilt Data Preview around five summary cards, chips, and a framed preview table
- rebuilt Schema Mapper into a left metadata column and right KPI configuration grid
- rebuilt Data Quality into summary cards plus a dedicated checks table
- polished Drift Alerts summary cards, filters, trend preview, and explanation panel
- rebuilt KPI Explorer into a filter rail, large chart panel, and bottom stat cards
- rebuilt Export into four large download cards and a local-generation callout
- cleaned up chart legends and current-window labeling

Accepted differences:
- Streamlit widget chrome and tab styling still impose some native spacing and control appearance differences
- iconography is approximated with text glyphs and card accents rather than a custom icon set
- table pagination remains Streamlit-native rather than fully custom

Final result: passed
