"""
Gene / protein ID mapping: ENSG <-> Symbol <-> UniProt.

Builds bidirectional maps between versioned ENSG IDs (RNA-seq),
gene symbols, and UniProt accession IDs (Olink proteomics) using
Ensembl BioMart, UniProt ID-Mapping API, and MyGene.info as fallbacks.

Functions
---------
build_ensg_to_symbol(df_deg, rna_feature_cols)
    Build ENSG -> gene symbol mapping from DEG file, BioMart, and mygene.
build_uniprot_to_ensg(uniprot_ids, symbol_to_ensg, rna_feature_cols)
    Build UniProt -> ENSG mapping via symbol bridge.
patch_unmapped_via_mygene(uniprot_ids, uniprot_to_symbol, uniprot_to_ensg,
                          symbol_to_ensg, rna_feature_cols)
    Patch remaining unmapped UniProt IDs via MyGene.info.
"""

import time
import numpy as np
import pandas as pd

from .config import OUTPUT_DIR


def pick_ensg(ens_field):
    """Extract best-effort unversioned ENSG from MyGene ensembl.gene field."""
    if ens_field is None:
        return None
    if isinstance(ens_field, float):
        return None
    if isinstance(ens_field, str):
        return ens_field.split(".")[0]
    if isinstance(ens_field, dict):
        g = ens_field.get("gene", None)
        if g:
            return str(g).split(".")[0]
    if isinstance(ens_field, (list, tuple)) and len(ens_field) > 0:
        first = ens_field[0]
        if isinstance(first, str):
            return first.split(".")[0]
        if isinstance(first, dict):
            g = first.get("gene", None)
            if g:
                return str(g).split(".")[0]
    return None


def build_ensg_to_symbol(df_deg, rna_feature_cols):
    """Build ENSG -> gene symbol mapping.

    Strategy: DEG file columns -> BioMart XML -> mygene fallback.

    Returns (ensg_to_symbol, symbol_to_ensg, all_mappings_cache).
    """
    import requests

    ensg_to_symbol = {}
    all_mappings = {}

    # Option 1: DEG file
    if any(c in df_deg.columns for c in ["gene_name", "symbol", "gene_symbol"]):
        symbol_col = [c for c in df_deg.columns
                      if c.lower() in ["gene_name", "symbol", "gene_symbol"]][0]
        ensg_to_symbol = dict(zip(df_deg["gene_id"], df_deg[symbol_col]))
        print(f"Loaded {len(ensg_to_symbol)} ENSG->Symbol from DEG file ({symbol_col})")
    else:
        print("No symbol column found in DEG file.")

    # Option 2: BioMart
    if len(ensg_to_symbol) < 100:
        print("\nFetching ENSG->Symbol mapping from Ensembl BioMart...")
        try:
            ensg_ids_stripped = list(set(eid.split(".")[0] for eid in rna_feature_cols))
            biomart_url = "http://www.ensembl.org/biomart/martservice"
            chunk_size = 500

            for i in range(0, len(ensg_ids_stripped), chunk_size):
                chunk = ensg_ids_stripped[i:i + chunk_size]
                id_filter = ",".join(chunk)
                xml_query = f'''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE Query>
<Query virtualSchemaName="default" formatter="TSV" header="1" uniqueRows="1">
  <Dataset name="hsapiens_gene_ensembl" interface="default">
    <Filter name="ensembl_gene_id" value="{id_filter}"/>
    <Attribute name="ensembl_gene_id"/>
    <Attribute name="external_gene_name"/>
    <Attribute name="uniprotswissprot"/>
  </Dataset>
</Query>'''
                resp = requests.get(biomart_url, params={"query": xml_query}, timeout=120)
                if resp.status_code == 200:
                    lines = resp.text.strip().split("\n")
                    for line in lines[1:]:
                        parts = line.split("\t")
                        if len(parts) >= 2 and parts[1]:
                            all_mappings[parts[0]] = {
                                "symbol": parts[1],
                                "uniprot": parts[2] if len(parts) > 2 else "",
                            }
                if (i // chunk_size) % 5 == 0:
                    print(f"  Processed {min(i + chunk_size, len(ensg_ids_stripped))}/{len(ensg_ids_stripped)}")

            for ensg_versioned in rna_feature_cols:
                ensg_base = ensg_versioned.split(".")[0]
                if ensg_base in all_mappings:
                    ensg_to_symbol[ensg_versioned] = all_mappings[ensg_base]["symbol"]

            print(f"BioMart mapping: {len(ensg_to_symbol)} ENSG->Symbol pairs")
        except Exception as e:
            print(f"BioMart fetch failed: {e}")

    # Option 3: mygene fallback
    if len(ensg_to_symbol) < 100:
        try:
            import mygene
            mg = mygene.MyGeneInfo()
            ensg_stripped = [eid.split(".")[0] for eid in rna_feature_cols[:5000]]
            results = mg.querymany(ensg_stripped, scopes="ensembl.gene",
                                   fields="symbol,uniprot", species="human",
                                   returnall=True)
            for hit in results["out"]:
                if "symbol" in hit and "query" in hit:
                    for ensg_v in rna_feature_cols:
                        if ensg_v.split(".")[0] == hit["query"]:
                            ensg_to_symbol[ensg_v] = hit["symbol"]
            print(f"mygene mapping: {len(ensg_to_symbol)} ENSG->Symbol pairs")
        except ImportError:
            print("mygene not installed. Install with: pip install mygene")
        except Exception as e:
            print(f"mygene failed: {e}")

    # Reverse mapping
    symbol_to_ensg = {}
    for ensg, sym in ensg_to_symbol.items():
        if sym:
            symbol_to_ensg.setdefault(sym, []).append(ensg)

    print(f"\nFinal mapping: {len(ensg_to_symbol)} ENSG->Symbol, "
          f"{len(symbol_to_ensg)} unique symbols")
    return ensg_to_symbol, symbol_to_ensg, all_mappings


def build_uniprot_to_ensg(uniprot_ids, symbol_to_ensg, rna_feature_cols,
                          all_mappings=None):
    """Build UniProt -> ENSG mapping via symbol bridge.

    Returns (uniprot_to_symbol, uniprot_to_ensg).
    """
    import requests

    uniprot_to_symbol = {}
    rna_cols_set = set(rna_feature_cols)

    # Step 1: Reverse BioMart cache
    if all_mappings:
        for ensg_base, info in all_mappings.items():
            sym = info.get("symbol", "")
            up = info.get("uniprot", "")
            if up and sym:
                for uid in up.split(";"):
                    uid = uid.strip()
                    if uid:
                        uniprot_to_symbol[uid] = sym
        print(f"Step 1 — reversed BioMart cache: {len(uniprot_to_symbol)} UniProt->Symbol")

    # Step 2: Dedicated BioMart query
    missing_uids = [u for u in uniprot_ids if u not in uniprot_to_symbol]
    if missing_uids:
        print(f"\nStep 2 — BioMart query for {len(missing_uids)} unmapped UniProt IDs ...")
        biomart_url = "http://www.ensembl.org/biomart/martservice"
        chunk_size = 300
        for i in range(0, len(missing_uids), chunk_size):
            chunk = missing_uids[i:i + chunk_size]
            id_str = ",".join(chunk)
            xml = (
                '<?xml version="1.0" encoding="UTF-8"?>'
                '<!DOCTYPE Query>'
                '<Query virtualSchemaName="default" formatter="TSV" header="1" uniqueRows="1">'
                '  <Dataset name="hsapiens_gene_ensembl" interface="default">'
                f'    <Filter name="uniprotswissprot" value="{id_str}"/>'
                '    <Attribute name="uniprotswissprot"/>'
                '    <Attribute name="external_gene_name"/>'
                '    <Attribute name="ensembl_gene_id"/>'
                '  </Dataset>'
                '</Query>'
            )
            try:
                resp = requests.get(biomart_url, params={"query": xml}, timeout=120)
                if resp.status_code == 200 and not resp.text.startswith("Query ERROR"):
                    for line in resp.text.strip().split("\n")[1:]:
                        parts = line.split("\t")
                        if len(parts) >= 2 and parts[0].strip() and parts[1].strip():
                            uniprot_to_symbol[parts[0].strip()] = parts[1].strip()
            except Exception as e:
                print(f"  Chunk {i // chunk_size} failed: {e}")
            time.sleep(0.5)

    # Step 3: UniProt ID-Mapping API
    still_missing = [u for u in uniprot_ids if u not in uniprot_to_symbol]
    if still_missing:
        print(f"\nStep 3 — UniProt ID-Mapping API for {len(still_missing)} remaining IDs ...")
        try:
            job_resp = requests.post(
                "https://rest.uniprot.org/idmapping/run",
                data={"from": "UniProtKB_AC-ID", "to": "Gene_Name",
                      "ids": ",".join(still_missing[:500])},
                timeout=30,
            )
            if job_resp.status_code == 200:
                job_id = job_resp.json()["jobId"]
                result_data = None
                for attempt in range(30):
                    time.sleep(2)
                    status = requests.get(
                        f"https://rest.uniprot.org/idmapping/status/{job_id}", timeout=30)
                    if status.status_code == 200:
                        sdata = status.json()
                        if "results" in sdata:
                            result_data = sdata
                            break
                        if sdata.get("jobStatus") == "FINISHED":
                            res = requests.get(
                                f"https://rest.uniprot.org/idmapping/results/{job_id}",
                                timeout=30)
                            result_data = res.json()
                            break
                if result_data:
                    for hit in result_data.get("results", []):
                        uid = hit.get("from", "")
                        sym = hit.get("to", "")
                        if uid and sym:
                            uniprot_to_symbol[uid] = sym
        except Exception as e:
            print(f"  Step 3 failed: {e}")

    # Step 4: Bridge UniProt -> Symbol -> ENSG
    uniprot_to_ensg = {}
    for uid in uniprot_ids:
        sym = uniprot_to_symbol.get(uid)
        if sym and sym in symbol_to_ensg:
            ensg_list = [e for e in symbol_to_ensg[sym] if e in rna_cols_set]
            if ensg_list:
                uniprot_to_ensg[uid] = ensg_list

    print(f"\nUniProt -> Symbol: {len(uniprot_to_symbol)}/{len(uniprot_ids)}")
    print(f"UniProt -> ENSG:   {len(uniprot_to_ensg)}/{len(uniprot_ids)}")
    return uniprot_to_symbol, uniprot_to_ensg


def patch_unmapped_via_mygene(uniprot_ids, uniprot_to_symbol, uniprot_to_ensg,
                              symbol_to_ensg, rna_feature_cols):
    """Patch remaining unmapped UniProt IDs via MyGene.info.

    Modifies uniprot_to_symbol and uniprot_to_ensg in place.
    """
    try:
        import mygene
    except ImportError:
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "mygene", "-q"])
        import mygene

    # Build base->versioned index
    ensg_base_to_versioned = {}
    for ensg_v in rna_feature_cols:
        base = ensg_v.split(".")[0]
        ensg_base_to_versioned.setdefault(base, []).append(ensg_v)

    def resolve_ensg_versioned(ensg_base):
        return ensg_base_to_versioned.get(ensg_base, [])

    still_unmapped = [u for u in uniprot_ids if u not in uniprot_to_symbol]
    print(f"UniProt IDs without symbol mapping: {len(still_unmapped)}/{len(uniprot_ids)}")

    mg = mygene.MyGeneInfo()

    if still_unmapped:
        chunk_size = 500
        n_patched_sym = 0
        n_patched_ensg_direct = 0
        n_patched_ensg_bridge = 0

        for i in range(0, len(still_unmapped), chunk_size):
            chunk = still_unmapped[i:i + chunk_size]
            try:
                hits = mg.querymany(
                    chunk, scopes="uniprot",
                    fields="symbol,ensembl.gene,ensembl.transcript,entrezgene",
                    species="human", as_dataframe=True, returnall=False,
                )
                if not isinstance(hits, pd.DataFrame) or len(hits) == 0:
                    continue

                hits = hits.reset_index().rename(columns={"query": "uid"})

                for _, row in hits.iterrows():
                    uid = row["uid"]
                    sym = row.get("symbol", None)
                    if isinstance(sym, str) and sym:
                        uniprot_to_symbol[uid] = sym
                        n_patched_sym += 1

                    ensg_base = pick_ensg(row.get("ensembl.gene", None))
                    if ensg_base and ensg_base.startswith("ENSG"):
                        versioned = resolve_ensg_versioned(ensg_base)
                        if versioned:
                            uniprot_to_ensg[uid] = versioned
                            n_patched_ensg_direct += 1
                            continue
                        else:
                            uniprot_to_ensg[uid] = [ensg_base]
                            n_patched_ensg_direct += 1
                            continue

                    sym_for_bridge = uniprot_to_symbol.get(uid)
                    if sym_for_bridge and sym_for_bridge in symbol_to_ensg:
                        versioned = [e for e in symbol_to_ensg[sym_for_bridge]
                                     if e in set(rna_feature_cols)]
                        if versioned:
                            uniprot_to_ensg[uid] = versioned
                            n_patched_ensg_bridge += 1
            except Exception as e:
                print(f"  MyGene chunk {i // chunk_size} failed: {e}")

        print(f"\nMyGene.info patch:")
        print(f"  New symbols:               {n_patched_sym}")
        print(f"  New ENSG (direct):         {n_patched_ensg_direct}")
        print(f"  New ENSG (symbol bridge):  {n_patched_ensg_bridge}")

    # Second pass: symbol bridge for IDs with symbol but no ENSG
    n_bridge_pass2 = 0
    for uid in uniprot_ids:
        if uid in uniprot_to_ensg:
            continue
        sym = uniprot_to_symbol.get(uid)
        if sym and sym in symbol_to_ensg:
            versioned = [e for e in symbol_to_ensg[sym] if e in set(rna_feature_cols)]
            if versioned:
                uniprot_to_ensg[uid] = versioned
                n_bridge_pass2 += 1
    print(f"  Symbol bridge pass 2:      {n_bridge_pass2}")

    # Save complete mapping
    patch_rows = []
    for uid in uniprot_ids:
        sym = uniprot_to_symbol.get(uid, "")
        ensg = uniprot_to_ensg.get(uid, [])
        ensg_str = ensg[0] if ensg else ""
        versioned = "yes" if ensg_str and "." in ensg_str else "no"
        patch_rows.append({
            "UniProt": uid, "Symbol": sym, "ENSG": ensg_str, "Versioned": versioned,
        })
    pd.DataFrame(patch_rows).to_csv(f"{OUTPUT_DIR}/uniprot_mapping_complete.csv", index=False)
    print(f"\nSaved: {OUTPUT_DIR}/uniprot_mapping_complete.csv")

    return uniprot_to_symbol, uniprot_to_ensg
