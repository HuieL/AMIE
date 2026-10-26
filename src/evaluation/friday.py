"""Gateway client for teacher, frozen-policy, and judge completions.

Reads FRIDAY_API_KEY / FRIDAY_BASE_URL from the repo-root .env. All calls are
temperature 0; empty or truncated completions are retried with an escalated
budget. Only the paper's model identifiers are accepted.
"""
import json
import time
import uuid
from pathlib import Path
import urllib.error
import urllib.request

class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs):return None

def friday_credentials():
    env={}
    path=Path(__file__).resolve().parents[2]/'.env'
    for line in path.read_text().splitlines():
        if '=' in line and not line.lstrip().startswith('#'):
            k,v=line.split('=',1);env[k.strip()]=v.strip().strip('\"\'')
    return env['FRIDAY_API_KEY'],env['FRIDAY_BASE_URL'].rstrip('/')

class FridayGenerator:
    def __init__(self,model):
        if model not in ('GPT-5.6-Sol','Qwen3-8B','GPT-5'):
            raise ValueError('Model has not passed an access probe')
        self.key,self.base=friday_credentials()
        self.model=model;self.opener=urllib.request.build_opener(NoRedirect())
        self.protocol={'model':model,'provider':'FRIDAY','revision':'not disclosed by service',
            'dtype':'not disclosed by service','temperature':0}

    def generate(self,conversations,max_tokens):
        answers=[];usage=[];traces=[];latencies=[]
        for messages in conversations:
            budget=max(1024,max_tokens)
            # Empty or truncated completions (finish_reason != 'stop') are
            # retried with an escalated budget.
            for budget_attempt in range(3):
                body={'model':self.model,'messages':messages,'temperature':0,'stream':False,
                      'max_tokens':budget}
                trace=uuid.uuid4().hex
                body['user']=trace
                started=time.monotonic()
                request=urllib.request.Request(self.base+'/chat/completions',data=json.dumps(body).encode(),
                    headers={'Authorization':'Bearer '+self.key,'Content-Type':'application/json','M-TraceId':trace})
                for attempt in range(3):
                    try:
                        with self.opener.open(request,timeout=180) as response:
                            result=json.load(response)
                            response_trace=response.headers.get('M-TraceId',trace)
                        break
                    except urllib.error.HTTPError as exc:
                        if exc.code in (408,429,500,502,503,504) and attempt < 2:
                            time.sleep(10*(attempt+1))
                            continue
                        raise RuntimeError(f'FRIDAY HTTP {exc.code}; trace {trace}') from None
                    except (urllib.error.URLError,TimeoutError):
                        if attempt < 2:
                            time.sleep(5*(attempt+1))
                            continue
                        raise RuntimeError(f'FRIDAY network failure; trace {trace}') from None
                if result.get('model')!=self.model:
                    raise ValueError('FRIDAY response model mismatch')
                choice=result['choices'][0];answer=choice['message'].get('content')
                if answer and choice.get('finish_reason')=='stop':
                    break
                if budget_attempt==2:
                    raise ValueError('Missing or truncated final answer')
                budget=min(8192,max(budget*2,4096))
            answers.append(answer);usage.append(result.get('usage',{}))
            traces.append(response_trace);latencies.append(time.monotonic()-started)
        return {'answers':answers,'protocol':self.protocol,'usage':usage,'trace_ids':traces,'latencies':latencies}
