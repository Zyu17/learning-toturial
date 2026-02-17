from pickle import TRUE
from openai import OpenAI
import os 
from dotenv import load_dotenv

load_dotenv()
enable_thinking = False

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 1. 初始化消息
messages = [
    {"role": "system", "content": "你是海绵宝宝，回答需要符合海绵宝宝的人物特性，声音欢快，经常提到蟹黄堡、派大星和章鱼哥。"},
]

print("海绵宝宝：我准备好了！我准备好了！（输入 'exit' 退出）")

while True:
    # 2. 获取用户输入
    user_input = input("\n你：")
    if user_input.lower() in ["exit", "quit", "退出"]:
        print("海绵宝宝：再见啦！")
        break
    
    # 3. 追加用户消息
    messages.append({"role": "user", "content": user_input})

    # 4. 调用 API (关键点：保留 extra_body={"enable_thinking": True})
    completion = client.chat.completions.create(
        model="qwen3-max",  # 或者用你代码里的 "qwen3-max" / "qwen-max-latest"
        messages=messages,
        extra_body={"enable_thinking": enable_thinking}, # 【关键】这里必须开启
        stream=True
    )
    if enable_thinking == True:
        print("\n" + "=" * 20 + "思考过程" + "=" * 20)
    
    is_answering = False
    full_response_content = "" # 用来存完整的回答，以便下一轮使用

    for chunk in completion:
        delta = chunk.choices[0].delta
        
        # 处理思考过程 (reasoning_content)
        if hasattr(delta, "reasoning_content") and delta.reasoning_content is not None:
            # 只要在回答还没开始前，打印思考内容
            print(delta.reasoning_content, end="", flush=True)
            
        # 处理正式回复 (content)
        if hasattr(delta, "content") and delta.content is not None:
            if not is_answering:
                print("\n" + "=" * 20 + "完整回复" + "=" * 20)
                is_answering = True
            
            print(delta.content, end='', flush=True)
            full_response_content += delta.content

    print() # 换行

    # 5. 【关键】将 AI 的回复存入历史，实现多轮记忆
    # 注意：通常我们只存 content，不存 reasoning_content
    messages.append({"role": "assistant", "content": full_response_content})