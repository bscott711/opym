import os
import json
import glob

def main():
    prof_dir = "/dev/shm/petakit_jobs/profiling"
    files = glob.glob(os.path.join(prof_dir, "*.json"))
    
    if not files:
        print("No profiling data found yet! Wait for jobs to finish.")
        return

    print("="*80)
    print(f"{'Job Name':<45} | {'Load':<6} | {'Decon':<6} | {'DSR':<6} | {'Save':<6} | {'Total':<6}")
    print("-" * 80)
    
    totals = {"load": 0, "decon": 0, "dsr": 0, "save": 0, "total": 0}
    count = 0
    
    for f in sorted(files):
        try:
            with open(f, 'r') as jf:
                data = json.load(jf)
                print(f"{data['job'][:43]:<45} | {data['load_s']:>5.1f}s | {data['decon_s']:>5.1f}s | {data['dsr_s']:>5.1f}s | {data['save_s']:>5.1f}s | {data['total_s']:>5.1f}s")
                totals["load"] += data["load_s"]
                totals["decon"] += data["decon_s"]
                totals["dsr"] += data["dsr_s"]
                totals["save"] += data["save_s"]
                totals["total"] += data["total_s"]
                count += 1
        except Exception as e:
            continue
            
    if count > 0:
        print("="*80)
        print(f"{'AVERAGE (' + str(count) + ' jobs)':<45} | {totals['load']/count:>5.1f}s | {totals['decon']/count:>5.1f}s | {totals['dsr']/count:>5.1f}s | {totals['save']/count:>5.1f}s | {totals['total']/count:>5.1f}s")
        print("="*80)

if __name__ == "__main__":
    main()
