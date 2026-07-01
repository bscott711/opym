import os
import glob
import re

def parse_time(time_str):
    try:
        return float(time_str.replace(' s', '').strip())
    except ValueError:
        return 0.0

def main():
    prof_dir = "/dev/shm/petakit_jobs/profiling"
    files = glob.glob(os.path.join(prof_dir, "*_html", "file0.html"))
    
    if not files:
        print("No HTML profiling data found yet! Wait for jobs to finish.")
        return

    print("="*85)
    print(f"{'Job Name':<45} | {'Load':<6} | {'Decon':<6} | {'DSR':<6} | {'Save':<6} | {'Total':<6}")
    print("-" * 85)
    
    totals = {"load": 0, "decon": 0, "dsr": 0, "save": 0, "total": 0}
    count = 0
    
    for f in sorted(files):
        job_name = os.path.basename(os.path.dirname(f)).replace('_html', '')
        
        try:
            with open(f, 'r', encoding='utf-8') as jf:
                content = jf.read()
            
            rows = re.findall(r'<tr.*?>(.*?)</tr>', content, re.IGNORECASE | re.DOTALL)
            
            load_s = 0.0
            decon_s = 0.0
            dsr_s = 0.0
            save_s = 0.0
            total_s = 0.0
            
            for row in rows:
                cols = re.findall(r'<td.*?>(.*?)</td>', row, re.IGNORECASE | re.DOTALL)
                if cols:
                    cols_text = [re.sub(r'<[^>]+>', '', c).strip() for c in cols]
                    if len(cols_text) < 3:
                        continue
                    
                    func_name = cols_text[0]
                    t_time = parse_time(cols_text[2])
                    
                    # Accumulate times
                    if func_name in ('readzarr', 'readtiff'):
                        load_s += t_time
                    elif func_name == 'decon_lucy_function':
                        decon_s += t_time
                    elif func_name == 'gpuArray.imwarp':
                        dsr_s += t_time
                    elif func_name == 'writezarr':
                        save_s += t_time
                    elif 'run_gpu_pipeline' in func_name and total_s == 0.0:
                        total_s = t_time  # Use main function as total if present
            
            if total_s == 0.0:
                total_s = load_s + decon_s + dsr_s + save_s
                
            print(f"{job_name[:43]:<45} | {load_s:>5.1f}s | {decon_s:>5.1f}s | {dsr_s:>5.1f}s | {save_s:>5.1f}s | {total_s:>5.1f}s")
            totals["load"] += load_s
            totals["decon"] += decon_s
            totals["dsr"] += dsr_s
            totals["save"] += save_s
            totals["total"] += total_s
            count += 1
            
        except Exception as e:
            continue
            
    if count > 0:
        print("="*85)
        print(f"{'AVERAGE (' + str(count) + ' jobs)':<45} | {totals['load']/count:>5.1f}s | {totals['decon']/count:>5.1f}s | {totals['dsr']/count:>5.1f}s | {totals['save']/count:>5.1f}s | {totals['total']/count:>5.1f}s")
        print("="*85)

if __name__ == "__main__":
    main()
