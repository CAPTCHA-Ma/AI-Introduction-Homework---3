mport torch
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

TEST_PROMPTS = [
    "请说出以下两句话区别在哪里？ 1、冬天：能穿多少穿多少 2、夏天：能穿多少穿多少",
    "请说出以下两句话区别在哪里？单身狗产生的原因有两个，一是谁都看不上，二是谁都看不上",
    "他知道我知道你知道他不知道吗？ 这句话里，到底谁不知道",
    "明明明明明白白白喜欢他，可她就是不说。 这句话里，明明和白白谁喜欢谁？",
    "领导：你这是什么意思？ 小明：没什么意思。意思意思。 领导：你这就不够意思了。 小明：小意思，小意思。领导：你这人真有意思。 小明：其实也没有别的意思。 领导：那我就不好意思了。 小明：是我不好意思。请问：以上“意思”分别是什么意思。"
]

def generate_response(model, tokenizer, prompt):
    messages = [{"role": "user", "content": prompt}]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=300,
            temperature=0.7,
            do_sample=True
        )
    
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

def main():
    parser = argparse.ArgumentParser(description="单次运行大模型测试脚本")
    parser.add_argument("--model_name", type=str, required=True, help="模型名称 (如: Qwen3-8B)")
    parser.add_argument("--model_path", type=str, required=True, help="模型本地路径 (如: /mnt/data/Qwen3-8B)")
    args = parser.parse_args()

    print("="*60)
    print(f"正在加载模型: {args.model_name}")
    print(f"本地路径: {args.model_path}")
    print("提示：从磁盘加载模型到显存大概需要1-3分钟，请观察下方的自带分块进度条...")
    print("="*60)
    
    try:

        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            trust_remote_code=True,
            torch_dtype="auto",
            device_map="auto"
        ).eval()
        
        print("\n模型加载至 GPU 完成！开始运行测试集...\n")
        
        for i, prompt in enumerate(tqdm(TEST_PROMPTS, desc=f"评测进度 ({args.model_name})", unit="题", colour="green")):

            tqdm.write(f"\n测试问题 {i+1}:\n{prompt}")
            
            response = generate_response(model, tokenizer, prompt)
            
            tqdm.write(f"\n回答:\n{response}")
            tqdm.write("-" * 60)
            
    except Exception as e:
        print(f"\n加载或运行 {args.model_name} 时发生错误: {e}")

if __name__ == "__main__":
    main()
