"""ML 가상환경의 실제 IPython 커널로 노트북 실행 및 출력 저장."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import json
from pathlib import Path
import sys
import tempfile
from jupyter_client import KernelManager

ROOT = Path(__file__).resolve().parents[1]
requested = Path(sys.argv[1] if len(sys.argv) > 1 else 'churn_llm_benchmark20.ipynb')
candidates = [requested] if requested.is_absolute() else [ROOT/requested, ROOT/'notebooks'/requested, ROOT/'notebooks/archive'/requested]
path = next((p for p in candidates if p.is_file()), None)
if path is None:
    raise FileNotFoundError('notebooks/ 또는 notebooks/archive/에서 노트북을 찾을 수 없습니다: '+str(requested))
notebook = json.loads(path.read_text(encoding='utf-8'))
with tempfile.TemporaryDirectory(prefix='churn_notebook_') as temporary:
    kernel = KernelManager(kernel_name='python3', connection_file=str(Path(temporary)/'kernel.json'))
    kernel.kernel_spec.argv = [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}']
    import os
    environment = dict(os.environ, IPYTHONDIR=str(Path(temporary)/'ipython'),
                       OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='2')
    kernel.start_kernel(cwd=str(ROOT), env=environment)
    client = kernel.blocking_client()
    client.start_channels()
    count = 0
    try:
        client.wait_for_ready(timeout=60)
        for cell in notebook['cells']:
            if cell['cell_type'] != 'code':
                continue
            message_id = client.execute(cell['source'], stop_on_error=True)
            outputs = []
            while True:
                message = client.get_iopub_msg(timeout=180)
                if message.get('parent_header', {}).get('msg_id') != message_id:
                    continue
                kind, content = message['header']['msg_type'], message['content']
                if kind == 'execute_input':
                    cell['execution_count'] = content['execution_count']
                elif kind == 'stream':
                    outputs.append(dict(output_type='stream', name=content['name'], text=content['text']))
                elif kind in ('display_data', 'execute_result'):
                    output = dict(output_type=kind, data=content['data'], metadata=content.get('metadata', {}))
                    if kind == 'execute_result':
                        output['execution_count'] = content['execution_count']
                    outputs.append(output)
                elif kind == 'error':
                    raise RuntimeError(content['ename'] + ': ' + content['evalue'])
                elif kind == 'status' and content['execution_state'] == 'idle':
                    break
            cell['outputs'] = outputs
            count += 1
            print('Executed', count, flush=True)
        notebook['metadata']['execution_note'] = 'Executed with actual ML kernel. See notebook settings for generation versus saved-result mode.'
        path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        print('PASS:', count, 'executed cells', flush=True)
    finally:
        client.stop_channels()
        kernel.shutdown_kernel(now=True)
