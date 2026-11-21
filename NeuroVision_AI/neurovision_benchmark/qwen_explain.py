"""Optional explanatory VLM, never used for real-time hazard decisions.

Usage: python qwen_explain.py --image /path/to/frame.png --evidence '{"camera_person_count":1,...}'
Requires separate pip install -r requirements-optional.txt and model weights.
"""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--image',type=Path,required=True)
    p.add_argument('--evidence',required=True,help='JSON of actual observations')
    args=p.parse_args()
    from PIL import Image
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    evidence=json.loads(args.evidence)
    if not isinstance(evidence,dict): raise ValueError('Evidence must be a JSON object')
    model_name='Qwen/Qwen3-VL-2B-Instruct'
    model=Qwen3VLForConditionalGeneration.from_pretrained(model_name,dtype='auto',device_map='auto')
    processor=AutoProcessor.from_pretrained(model_name)
    image=Image.open(args.image).convert('RGB')
    prompt=("Descrie concis în română doar obiectele vizibile și dovezile senzorilor. "
       "Nu afirma că Wi-Fi a identificat persoana din cameră. Nu inventa obiecte, "
       "distanțe sau persoane invizibile. Evidențiază incertitudinea. "
       "Dovezi structurate:\n"+json.dumps(evidence,ensure_ascii=False))
    messages=[{'role':'user','content':[{'type':'image','image':image},
                                         {'type':'text','text':prompt}]}]
    inputs=processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,
                                         return_dict=True,return_tensors='pt').to(model.device)
    output=model.generate(**inputs,max_new_tokens=160,do_sample=False)
    text=processor.batch_decode([o[len(i):] for i,o in zip(inputs.input_ids,output)],
                                skip_special_tokens=True)[0]
    print(text)
    print('\nWARNING: unverified model-generated text; manually check against evidence.')

if __name__=='__main__': main()
