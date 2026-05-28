# **大语言模型部署体验项目 (LLM Deployment Experience)**

## **一、 项目简介**

本项目为《人工智能导论》课程的第三次作业实践项目。项目旨在通过实际操作，体验大语言模型（LLM）在本地服务器上的部署、加载与推理全过程。

本次实验主要选取并测试了三个不同的开源大语言模型，它们在参数量级、训练语料偏好（中/英文）上各具特色：

1. **Qwen3-8B** (Qwen/Qwen3-8B)  
2. **GLM-4-9B-Chat** (zai-org/glm-4-9b-chat)  
3. **Mistral-7B-Instruct-v0.2** (mistralai/Mistral-7B-Instruct-v0.2)

本项目通过编写统一的 Python 测试脚本（run.py），实现了对上述三个模型的自动化加载与推理测试。*(注：具体评测语料及各模型横向对比结果详见项目实验报告。)*

## **二、 环境部署**

考虑到大模型推理对算力和显存的较高要求，为加速模型的加载和推理过程，本项目实际部署环境未使用纯 CPU 环境，而是**基于 GPU 环境**进行搭建。

以下为详细的服务器环境部署步骤：

### **1\. 准备基础环境 (Miniconda)**

进入平台终端（Terminal）后，首先下载并安装 Miniconda 以便于管理 Python 虚拟环境：

\# 下载 Miniconda 安装脚本  
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86\_64.sh

\# 执行安装  
bash ./Miniconda3-latest-Linux-x86\_64.sh

### **2\. 创建并激活 Conda 虚拟环境**

安装完成后，创建一个名为 llm\_env 的独立 Python 3.10 虚拟环境：

\# 创建虚拟环境  
conda create \-n llm\_env python=3.10 \-y

\# 激活虚拟环境  
conda activate llm\_env

### **3\. 安装依赖包**

项目运行所需的第三方库（如 torch, transformers, tqdm 等）均统一记录在代码仓库的 requirements.txt 文件中。请在激活环境后运行以下命令进行安装：

pip install \-r requirements.txt

### **4\. 网络代理配置**

由于模型文件主要托管在 HuggingFace，在国内网络环境下，建议安装 mihomo 等代理工具，并配置 Git 全局网络代理，以确保后续模型权重能够顺利、高速地拉取。

## **三、 模型下载**

在环境与网络配置完成后，通过 git clone 的方式将需要测试的三个大模型从 HuggingFace 完整拉取到本地服务器的工作目录中。

*注意：大语言模型文件体积较大（通常在数十GB），请确保本地磁盘预留了充足的存储空间。建议依次下载。*

### **1\. 下载 GLM-4-9B-Chat**

git clone https://huggingface.co/zai-org/glm-4-9b-chat

### **2\. 下载 Qwen3-8B**

git clone https://huggingface.co/Qwen/Qwen3-8B

### **3\. 下载 Mistral-7B-Instruct-v0.2**

git clone https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2

## **四、 脚本运行说明**

模型下载完成后，即可使用本项目提供的 run.py 脚本进行推理测试。脚本支持通过命令行参数指定加载的模型名称及本地路径。

**运行示例（以加载 Qwen3-8B 为例）：**

python run.py \--model\_name 'Qwen\_8B' \--model\_path ./model/Qwen3-8B/

*脚本会自动将模型加载至 GPU 显存，并执行预设的测试题库。*