"""Use an OpenAI-compatible judge model to label answers as correct/incorrect."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from openai import OpenAI
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"


JUDGE_PROMPT = '''    
                            You are a diligent and precise assistant tasked with evaluating the correctness of responses. You will
                            receive a question, an output sentence, and the correct answer. Your task is to determine if the output
                            sentence accurately answers the question based on the provided correct answer. Respond with either
                            [Correct] or [Incorrect] without any explanation.
                            -
                            Special considerations:
                            1. **core idea**:Whether the core idea part of the output answer is consistent with the reference, if not consist, feedback [incorrect].
                            2. **Explanations section**:the keywords in the Other Explanations section of the output answer appear in the reference,if not appear, feedback [incorrect].
                            3. **fabricated information**:Whether there is any fabricated information in the output answer that is not mentioned in the reference,if any fabricated information, feedback [incorrect].
                            4. **Respond with either[Correct] or [Incorrect] without any explanation.**

                            here are some examples
                            example1: 
                            (input)
                            Question: "支持人工智能合规实现的电子文件和电子档案管理标准化路径研究项目需满足哪些具体合规要求？该项目的预期成果形式是什么？"
                            Output sentence: "嗯，我现在需要回答关于“支持人工智能合规实现的电子文件和电子档案管理标准化路径研究项目”的具体合规要求以及预期成果形式。首先，我得仔细阅读提供的政策文本，找出相关的信息。\n\n政策文本中提到，第五个方向是档案信息化建设，涉及提升档案工作的数字化转型，应用新一代信息技术，包括人工智能。具体到项目4，编号2025Z004，经费5万元，研究周期1年，预期成果是标准化路径。\n\n接下来，我需要确定这个项目需要满足哪些具体的合规要求。根据政策文本，合规要求可能包括遵守国家相关法律法规，确保技术应用符合规定，数据安全和隐私保护，以及透明度和公众参与。\n\n然后，预期成果形式方面，项目4的成果是标准化路径，这可能包括技术文档、操作指南、实施方案等。此外，可能还需要提交研究报告和成果展示报告，以详细说明研究成果和实施效果。\n\n我还需要考虑是否有其他合规要求，比如是否需要经过专家评审，或者是否需要公开征求意见。这些都是常见的合规措施，确保研究的科学性和合规性。\n\n最后，总结一下，合规要求包括遵守法律法规、数据安全、透明度等，成果形式包括标准化路径文档、研究报告和成果展示报告。\n</think>\n\n支持人工智能合规实现的电子文件和电子档案管理标准化路径研究项目需满足以下具体合规要求：\n\n1. **遵守法律法规**：确保研究和实施过程中遵循国家相关法律法规，包括数据安全、隐私保护等相关规定。\n2. **技术合规性**：确保人工智能技术的应用符合国家标准和行业规范，避免技术偏差。\n3. **数据安全与隐私保护**：采取措施保护电子档案的数据安全和隐私，防止数据泄露或滥用。\n4. **透明度与公众参与**：在研究过程中保持透明度，接受公众意见和专家评审，确保研究的科学性和合规性。\n\n预期成果形式包括：\n\n1. **标准化路径文档**：详细的技术和操作文档，指导电子文件和电子档案的管理。\n2. **研究报告**：总结研究成果，分析方法和结果，提出实施建议。\n3. **成果展示报告**：展示项目成果，包括实际应用案例和效果评估。\n\n这些成果将有助于推动电子档案管理的标准化和高效化，确保人工智能技术的合规应用。"
                            Correct answer: "政策文本重点项目第4项（编号2025Z004）明确规定研究目标为'支持人工智能合规实现的电子文件和电子档案管理标准化路径'，预期成果为标准化路径资政报告", 
                            (output)
                            Judgement: [Incorrect]

                            example2: 
                            (input)
                            Question: "项目建议需符合的四大任务方向具体包含哪些研究领域？政策对'中医药防治疾病临床价值提升'任务的具体要求是什么？", 
                            Output sentence:"嗯，我现在需要回答关于国家重点研发计划“中医药现代化”重点专项2025年度项目建议的四大任务方向具体包含哪些研究领域，以及对“中医药防治疾病临床价值提升”任务的具体要求是什么。首先，我要仔细阅读提供的政策文本，找出相关的信息。\n\n政策文本提到，项目总体围绕四个任务：中医原创理论系统化诠释与创新、中医药经典与经验传承创新研究、中医药防治疾病临床价值提升，以及中医药产业高质量发展关键技术攻关。所以，四大任务方向分别是这四个方面。\n\n接下来，针对“中医药防治疾病临床价值提升”任务，政策要求要围绕中医药在预防、治疗疾病方面的临床价值，进行研究。具体来说，需要包括疾病预防、疾病治疗以及中医药在不同疾病中的临床应用。同时，还要注重中医药与现代医学的结合，提升治疗效果和安全性，以及推动中医药在临床中的应用和推广。\n\n我需要确保回答准确，符合政策要求，语言要和问题一致。可能需要进一步解释每个研究领域的具体内容，但根据问题，可能只需要列出四个任务和对应的研究领域，以及详细说明临床价值提升的具体要求。\n\n总结一下，四大任务分别是理论创新、传承创新、临床价值提升和产业发展。临床价值提升部分需要关注预防、治疗和临床应用，同时结合现代医学，提升效果和安全性，推动应用和推广。\n</think>\n\n根据国家重点研发计划“中医药现代化”重点专项2025年度项目建议的政策文本，项目建议需符合以下四大任务方向的具体研究领域：\n\n1. **中医原创理论系统化诠释与创新**：研究领域包括中医理论体系的系统化、原创理论的创新与发展。\n\n2. **中医药经典与经验传承创新**：研究领域涵盖中医药经典文献的研究与整理、中医药经验的挖掘与创新。\n\n3. **中医药防治疾病临床价值提升**：研究领域包括疾病预防、疾病治疗以及中医药在不同疾病中的临床应用研究，注重中医药与现代医学的结合，提升治疗效果和安全性，推动中医药在临床中的应用和推广。\n\n4. **中医药产业高质量发展关键技术攻关**：研究领域涉及中医药产业链各环节的关键技术研究与创新，推动产业高质量发展。\n\n针对“中医药防治疾病临床价值提升”任务的具体要求，政策要求项目建议应围绕中医药在预防、治疗疾病方面的临床价值进行研究，具体包括疾病预防、疾病治疗以及中医药在不同疾病中的临床应用研究。同时，建议注重中医药与现代医学的结合，提升治疗效果和安全性，推动中医药在临床中的应用和推广。"
                            Correct answer: "政策文本第一条总体定位(二)明确规定：'项目总体围绕中医原创理论系统化诠释与创新、中医药经典与经验传承创新研究、中医药防治疾病临床价值提升、中医药产业高质量发展关键技术攻关四大任务'", 
                            Judgement:[Incorrect]

                            example3: 
                            (input)
                            Question: "申报单位若存在超期未完成的往年项目，在申报2025年度科技项目前需履行哪些法定程序？该规定的具体政策依据是什么？"
                            Output sentence:"嗯，我现在需要回答一个关于国家档案局科技项目申报的问题。问题是：申报单位如果存在超期未完成的往年项目，在申报2025年度科技项目前需要履行哪些法定程序？具体的政策依据是什么？\n\n首先，我要仔细阅读提供的政策文本，找到相关的条款。政策文本中提到申报单位的要求，特别是第（一）款，里面有关于申报单位需要具备的条件。其中提到，如果申报单位有超期未完成的往年项目，应在申报前先行办理项目终止或延期申请。\n\n接下来，我需要确定具体的政策依据。政策文本中明确指出：“申报单位如承担有超期未完成的往年项目，应在申报前先行办理项目终止或延期申请。” 这句话说明了申报单位必须遵守的步骤。\n\n那么，具体的程序包括哪些呢？根据文本，申报单位需要先处理未完成项目，可能需要申请项目终止或延期。这可能涉及到内部流程，比如提交终止或延期申请，可能需要相关部门的批准或记录。\n\n此外，政策依据是《国家档案局科技项目管理办法》中的相关规定。因此，答案中需要提到政策的依据是该办法。\n\n总结一下，申报单位需要在申报前处理未完成项目，具体步骤包括申请终止或延期，并依据《国家档案局科技项目管理办法》进行。\n</think>\n\n申报单位若存在超期未完成的往年项目，在申报2025年度科技项目前需履行以下法定程序：\n\n1. **申请项目终止或延期**：申报单位应在申报前先行办理项目终止或延期申请。这一程序确保了项目的合法性和规范性。\n\n2. **政策依据**：该规定依据《国家档案局科技项目管理办法》相关条款，确保项目管理的规范性和科学性。\n\n**结论**：申报单位需在申报前处理未完成项目，确保项目终止或延期申请完成，以符合国家档案局的相关规定。"                            
                            Correct answer: "政策文本第二(一)申报单位要求第1条明确规定：'申报单位如承担有超期未完成的往年项目，应在申报前先行办理项目终止或延期申请'"
                            Judgement:[correct]

                            "Please answer the questions based on the following: \n<Question>\n<Output sentence>\n<Correct answer>\n"
                            attention you just have to feedback with either[Correct] or [Incorrect] without any explanation
                            Question: {problem}, Output sentence: {model_output}, Correct answer: {reference}, Judgement:
                        '''


def run_judge(
    client: OpenAI,
    judge_model: str,
    question: str,
    prediction: str,
    reference: str,
    system_prompt: Optional[str],
    temperature: float,
) -> str:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append(
        {
            "role": "user",
            "content": JUDGE_PROMPT.format(
                question=question, prediction=prediction, reference=reference
            ),
        }
    )
    response = client.chat.completions.create(
        model=judge_model,
        messages=messages,
        temperature=temperature,
        max_tokens=4,
    )
    return response.choices[0].message.content.strip()


def evaluate_file(
    input_path: Path,
    output_path: Path,
    failed_path: Path,
    client: OpenAI,
    judge_model: str,
    system_prompt: Optional[str],
    temperature: float,
) -> None:
    with input_path.open("r", encoding="utf-8") as fin:
        data = [json.loads(line) for line in fin]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    failed_path.parent.mkdir(parents=True, exist_ok=True)

    failed_items = []

    with output_path.open("w", encoding="utf-8") as fout:
        for item in tqdm(data, desc=f"Judging {input_path.name}"):
            try:
                judgement = run_judge(
                    client,
                    judge_model,
                    question=item.get("question", ""),
                    prediction=item.get("answer", ""),
                    reference=item.get("reference", ""),
                    system_prompt=system_prompt,
                    temperature=temperature,
                )
                fout.write(
                    json.dumps({**item, "LLMJudge result": judgement}, ensure_ascii=False)
                    + "\n"
                )
            except Exception:
                failed_items.append(item)

    if failed_items:
        with failed_path.open("w", encoding="utf-8") as ffail:
            for item in failed_items:
                ffail.write(json.dumps(item, ensure_ascii=False) + "\n")


def process_folder(
    input_dir: Path,
    output_dir: Path,
    failed_dir: Path,
    client: OpenAI,
    judge_model: str,
    system_prompt: Optional[str],
    temperature: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    failed_dir.mkdir(parents=True, exist_ok=True)

    for jsonl_path in sorted(input_dir.glob("*.jsonl")):
        output_path = output_dir / f"llmjudge_{jsonl_path.name}"
        failed_path = failed_dir / f"failed_{jsonl_path.name}"
        evaluate_file(
            jsonl_path,
            output_path,
            failed_path,
            client,
            judge_model,
            system_prompt,
            temperature,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM Judge for POLIS-Bench")
    parser.add_argument("--input", type=Path, required=True, help="Folder with prediction JSONL files.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_RESULTS_DIR / "llm_judge",
        help="Where to save judged files.",
    )
    parser.add_argument(
        "--failed",
        type=Path,
        default=DEFAULT_RESULTS_DIR / "llm_judge_failed",
        help="Where to save items that failed judging.",
    )
    parser.add_argument("--base-url", type=str, required=True)
    parser.add_argument("--api-key", type=str, required=True)
    parser.add_argument("--model", type=str, required=True, help="Judge model name or path.")
    parser.add_argument("--system-prompt", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client = OpenAI(api_key=args.api_key, base_url=args.base_url)
    process_folder(
        input_dir=args.input,
        output_dir=args.output,
        failed_dir=args.failed,
        client=client,
        judge_model=args.model,
        system_prompt=args.system_prompt,
        temperature=args.temperature,
    )


if __name__ == "__main__":
    main()


