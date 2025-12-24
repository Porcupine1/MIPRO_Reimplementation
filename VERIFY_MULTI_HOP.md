# How to Verify Multi-Hop Retrieval is Working

**Date:** December 23, 2025

---

## Quick Answer

Multi-hop retrieval is working if you see:
1. ✅ **hop1 titles** in the DEBUG logs
2. ✅ **bridge titles** in the DEBUG logs  
3. ✅ **hop2 titles** in the DEBUG logs
4. ✅ **Different Wikipedia pages** in the retrieved context (not just from hop1)

---

## Method 1: Enable DEBUG Logging (Recommended)

### Step 1: Enable DEBUG logging

I've already changed `main.py` line 143 to use `logging.DEBUG` instead of `logging.INFO`.

### Step 2: Run with cache

```bash
uv run python main.py --tier light --use-cache
```

### Step 3: Look for these log lines

In `outputs/run.log` or console output, search for:

```
[DEBUG] retrievers.wikipedia_api: WikipediaRetriever: hop1 titles: ['Title1', 'Title2', ...]
[DEBUG] retrievers.wikipedia_api: WikipediaRetriever: bridge titles: ['BridgeTitle1', ...]
[DEBUG] retrievers.wikipedia_api: WikipediaRetriever: returning N passages total
```

**Example of working multi-hop:**
```
2025-12-23 16:44:00,520 [DEBUG] retrievers.wikipedia_api: WikipediaRetriever: starting retrieval for question: When was the defending titlist of 2009–10 Biathlon World Cup – Pursuit Men born?
2025-12-23 16:44:01,234 [DEBUG] retrievers.wikipedia_api: WikipediaRetriever: hop1 titles: ['Ole Einar Bjørndalen', '2009–10 Biathlon World Cup']
2025-12-23 16:44:02,456 [DEBUG] retrievers.wikipedia_api: Selected bridge titles: ['Biathlon', 'Norway national biathlon team']
2025-12-23 16:44:03,789 [DEBUG] retrievers.wikipedia_api: WikipediaRetriever: returning 6 passages total
```

---

## Method 2: Check Retrieved Context

### Look at the RETRIEVED CONTEXT section

In the logs, you'll see:

```
[INSTR] programs.QAProgram: RETRIEVED CONTEXT (6734 chars, 46 lines):
[INSTR] programs.QAProgram: [Hugh Laurie | ...]
[INSTR] programs.QAProgram: [Blackadder | ...]  ← Different page (hop2)
[INSTR] programs.QAProgram: [Stephen Fry | ...]  ← Different page (hop2)
```

**Signs of multi-hop working:**
- ✅ Multiple different Wikipedia page titles in `[Title | ...]` format
- ✅ Pages that weren't in the original query but are related through links
- ✅ More than 2-3 different page titles

**Signs of single-hop only:**
- ❌ Only 1-2 page titles
- ❌ All sentences from the same article
- ❌ Missing relevant information that would be on linked pages

---

## Method 3: Check Configuration

Verify your config has multi-hop enabled:

```bash
grep -A 5 "HOPS\|MAX_WIKI_TITLES_TOTAL" config.py
```

**Expected output:**
```python
HOPS = 2  # Must be 2 for multi-hop
MAX_WIKI_TITLES_TOTAL = 10  # Must be > 4 to allow hop2 titles
```

**If you see:**
```python
HOPS = 1  # ❌ Single-hop only
MAX_WIKI_TITLES_TOTAL = 4  # ❌ Too small, blocks hop2
```

Then multi-hop is disabled.

---

## Method 4: Examine the Retrieval Flow

### Understanding the Multi-Hop Process

From `retrievers/wikipedia_api.py`:

```python
def retrieve(self, question, query, example):
    # Step 1: Hop 1 - Direct search
    hop1_titles = self._search(seed_query, self.top_titles_hop1)
    logger.debug("WikipediaRetriever: hop1 titles: %s", hop1_titles)
    
    # Step 2: Extract bridge titles from hop1 links
    bridge_titles = self._extract_bridge_titles(question, hop1_titles)
    logger.debug("WikipediaRetriever: bridge titles: %s", bridge_titles)
    
    # Step 3: Hop 2 - Search using bridge titles
    hop2_titles = [
        self._search(f"{question} {title}", self.top_titles_hop2)
        for title in bridge_titles
    ]
    
    # Step 4: Merge and deduplicate
    merged_titles = hop1_titles + hop2_titles
    passages = self._fetch_passages_for_titles(merged_titles)
```

**Key indicators:**
1. **hop1_titles:** Direct search results (e.g., "Ole Einar Bjørndalen")
2. **bridge_titles:** Linked pages from hop1 (e.g., "Biathlon", "Norway")
3. **hop2_titles:** Secondary search results using bridges
4. **merged_titles:** Combined list (should be > hop1_titles count)

---

## Method 5: Count Unique Page Titles

### Simple grep command

```bash
# Run your program and save output
uv run python main.py --tier light --use-cache 2>&1 | tee test_output.log

# Count unique Wikipedia page titles in context
grep -o '\[.*|' test_output.log | sort -u | wc -l
```

**Expected:**
- **Single-hop:** 2-4 unique titles
- **Multi-hop:** 5-10+ unique titles

---

## Example: Working Multi-Hop

### Question
"When was the defending titlist of 2009–10 Biathlon World Cup – Pursuit Men born?"

### Expected Flow

**Hop 1 Search:** "Ole Einar Bjørndalen birthdate"
- Finds: "Ole Einar Bjørndalen", "2009–10 Biathlon World Cup"

**Bridge Extraction:** Links from those pages
- From "Ole Einar Bjørndalen" → "Biathlon", "Norway national biathlon team"
- From "2009–10 Biathlon World Cup" → "Biathlon World Cup", "Pursuit (biathlon)"

**Hop 2 Search:** Using bridges
- "When was... Biathlon" → More biathlon pages
- "When was... Norway national biathlon team" → Team info pages

**Final Context:** 6-8 different Wikipedia pages merged

### Debug Logs to Look For

```
[DEBUG] retrievers.wikipedia_api: WikipediaRetriever: hop1 titles: ['Ole Einar Bjørndalen', '2009–10 Biathlon World Cup']
[DEBUG] retrievers.wikipedia_api: Selected bridge titles: ['Biathlon', 'Norway national biathlon team']
[DEBUG] retrievers.wikipedia_api: WikipediaRetriever: returning 6 passages total
```

---

## Troubleshooting

### Issue: Only seeing hop1 titles, no bridge titles

**Possible causes:**
1. `HOPS = 1` in config (check `config.py`)
2. No links found on hop1 pages (rare)
3. No lexical overlap between links and question

**Fix:**
```python
# In config.py
HOPS = 2  # Enable multi-hop
```

### Issue: Bridge titles found but no hop2 titles

**Possible causes:**
1. `MAX_WIKI_TITLES_TOTAL = 4` (too small)
2. `TOP_TITLES_HOP2 = 0` (disabled)

**Fix:**
```python
# In config.py
MAX_WIKI_TITLES_TOTAL = 10  # Allow room for hop2
# In tier config
top_titles_hop2=2  # Or higher
```

### Issue: Not seeing DEBUG logs

**Possible causes:**
1. Logging level is INFO or higher
2. Looking at wrong log file

**Fix:**
```python
# In main.py line 143
setup_logging(
    level=logging.DEBUG,  # Changed from INFO
    ...
)
```

Or check the file:
```bash
tail -f outputs/run.log | grep -E "hop1|hop2|bridge"
```

---

## Current Configuration Status

Based on your config:

```python
# config.py
HOPS = 2  ✅ Multi-hop enabled
MAX_WIKI_TITLES_TOTAL = 10  ✅ Allows hop2 titles

# LIGHT_CONFIG
top_titles_hop1=2  ✅ Gets 2 hop1 titles
top_titles_hop2=2  ✅ Gets 2 hop2 titles per bridge
```

**Expected behavior:**
- Hop1: 2 titles
- Bridge: 1-3 titles (from hop1 links)
- Hop2: 2-6 titles (2 per bridge)
- **Total: 5-11 unique titles** in final context

---

## Quick Verification Script

Create a test script to verify multi-hop:

```bash
#!/bin/bash
# verify_multihop.sh

echo "Running test with DEBUG logging..."
uv run python main.py --tier light --use-cache 2>&1 | tee /tmp/test.log

echo ""
echo "=== Multi-Hop Verification ==="
echo ""

# Check for hop1
if grep -q "hop1 titles:" /tmp/test.log; then
    echo "✅ Hop1 titles found:"
    grep "hop1 titles:" /tmp/test.log | head -3
else
    echo "❌ No hop1 titles found"
fi

echo ""

# Check for bridge
if grep -q "bridge titles:" /tmp/test.log; then
    echo "✅ Bridge titles found:"
    grep "bridge titles:" /tmp/test.log | head -3
else
    echo "❌ No bridge titles found (might be single-hop only)"
fi

echo ""

# Count unique pages
unique_pages=$(grep -o '\[.*|' /tmp/test.log | sort -u | wc -l)
echo "📊 Unique Wikipedia pages in context: $unique_pages"

if [ "$unique_pages" -gt 4 ]; then
    echo "✅ Multi-hop appears to be working ($unique_pages > 4 pages)"
else
    echo "⚠️  Might be single-hop only ($unique_pages ≤ 4 pages)"
fi
```

---

## Summary

**To verify multi-hop is working:**

1. ✅ Enable DEBUG logging (already done in main.py)
2. ✅ Run: `uv run python main.py --tier light --use-cache`
3. ✅ Check logs for: `hop1 titles`, `bridge titles`, `hop2 titles`
4. ✅ Verify: Multiple different Wikipedia pages in RETRIEVED CONTEXT
5. ✅ Count: Should see 5-10+ unique page titles

**Key log lines to search for:**
```bash
grep -E "hop1 titles|bridge titles|returning.*passages" outputs/run.log
```

If you see all three types of log lines, multi-hop retrieval is working! 🎯

---

**Created:** December 23, 2025  
**Status:** ✅ DEBUG logging enabled, ready to verify

