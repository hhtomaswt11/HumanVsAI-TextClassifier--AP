import os
import json
from dataclasses import dataclass
from typing import Optional
from datetime import datetime
import google.generativeai as genai


@dataclass
class ClassificationResult:
    text: str
    classification: str  # "Human", "OpenAI", "Meta", "Google", "Anthropic"
    confidence: float  # 0-100
    reasoning: str
    tokens_used: int


class AISourceClassifier:
    
    def __init__(self, model: str = "gemma-3-27b-it"):
        genai.configure(api_key="AIzaSyBT1hLYKjbLVrPAAadFwBgX8q_TmtcaIf0")
        self.model = genai.GenerativeModel(model)
        
        
        self.examples = {
            "Human": [
                # Human-written scientific definition 1: More casual, varied structure
                """Osmosis is the movement of water molecules across a semipermeable membrane from areas 
where water is more concentrated to areas where it's less concentrated. This happens because water molecules 
are constantly moving around, and they tend to move toward regions where there's more dissolved stuff, like salt 
or sugar. The water isn't actually attracted to the dissolved particles—it's just random motion that creates an 
overall drift. This process is important in cells and is why putting salt on a wound can draw fluid out of nearby 
tissue.""",
                
                # Human-written scientific definition 2: Less formal, some hedging
                """Enthalpy is basically the total heat content of a system. When we talk about chemical reactions, 
we usually care about how much heat is released or absorbed, which is the change in enthalpy. It's denoted as H, 
and chemists often think of it as similar to energy, though technically it includes pressure and volume information 
too. In everyday chemistry, we often use it almost interchangeably with energy because we're usually working at constant 
pressure. The key is that positive enthalpy means the reaction absorbs heat, while negative means it releases heat.""",
                
                # Human-written scientific definition 3: Variable terminology
                """The greenhouse effect happens when certain gases in the atmosphere trap heat from the sun. 
Carbon dioxide, methane, and water vapor are the main culprits. These gases let sunlight through pretty easily, 
but when that light gets reflected as heat from Earth's surface, the gases block a lot of it from escaping back 
into space. It's kind of like how a blanket keeps you warm, or how a car heats up inside on a sunny day. Without 
this effect, Earth would be too cold for most life, but right now we have too much of it going on.""",
            ],
            
            "OpenAI": [
                # OpenAI style: Very clear, confident, structured, somewhat formal
                """Photosynthesis is the biochemical process by which plants, algae, and certain bacteria convert 
light energy into chemical energy stored in glucose. This process occurs in two main stages: light-dependent reactions 
that occur in the thylakoid membranes of chloroplasts, where light energy excites electrons and splits water molecules 
to produce ATP and NADPH; and light-independent reactions (the Calvin cycle) occurring in the stroma, which use ATP 
and NADPH to fix carbon dioxide into glucose. This mechanism is fundamental to most life on Earth.""",
                
                # OpenAI style: Clear definitions, good structure, confident tone
                """Catalysis is the process of speeding up a chemical reaction by introducing a catalyst, a substance 
that participates in the reaction but remains unchanged at the end. Catalysts work by lowering the activation energy 
required for the reaction to proceed, allowing reactants to form products more quickly without being consumed themselves. 
This mechanism is crucial in industrial chemistry, biological systems, and environmental applications. Catalysts enable 
reactions to occur under milder conditions, improving efficiency and reducing energy consumption in manufacturing processes.""",
                
                # OpenAI style: Precise, pedagogical structure
                """Diffusion is the spontaneous movement of particles from regions of higher concentration to regions 
of lower concentration, driven by the random thermal motion of molecules. Unlike active transport, diffusion requires 
no energy input and continues until equilibrium is reached. The rate of diffusion depends on factors including the 
concentration gradient, temperature, particle size, and the medium through which diffusion occurs. This fundamental 
process is essential for gas exchange in living organisms and plays a key role in numerous chemical and biological phenomena.""",
            ],
            
            "Meta": [
                # Meta style: Clear but sometimes slightly verbose, practical examples
                """Viscosity describes how resistant a fluid is to flowing. Imagine honey compared to water—honey flows 
much more slowly because it has higher viscosity. This resistance comes from internal friction between the molecules 
in the liquid as they move past each other. Different liquids have different viscosities depending on their molecular 
structure and temperature. For example, hot honey flows more easily than cold honey because increasing temperature gives 
molecules more energy to overcome the internal friction. Viscosity is important in many applications, from designing 
lubricants for engines to understanding how blood flows through our veins.""",
                
                # Meta style: Accessible, includes context and examples
                """Electrochemistry is the branch of chemistry that deals with the relationship between electrical current 
and chemical reactions. When you have a battery, for example, chemical reactions inside generate electrical current that 
flows through an external circuit. Conversely, if you apply electrical current to certain substances, you can drive chemical 
reactions that wouldn't happen naturally. This is called electrolysis and is used in many industrial processes like metal 
refining and water treatment. The fundamental principle is that electrons transfer between chemical species, creating the 
connection between chemistry and electricity.""",
                
                # Meta style: Informative, somewhat accessible tone
                """Torque is a measure of how much a force causes an object to rotate about an axis. Think of tightening 
a bolt with a wrench—the further your hand is from the bolt, the easier it is to turn it, even with the same amount of force. 
This is because torque depends not just on the force applied, but also on the distance from the axis of rotation, called the 
moment arm. Mathematically, torque equals force times the perpendicular distance. Understanding torque is essential in engineering, 
mechanics, and understanding how machines like motors, turbines, and gears work.""",
            ],
            
            "Google": [
                # Google style: Clear, well-structured, slightly formal but accessible
                """Entropy is a measure of disorder or randomness in a system, and it tends to increase over time in isolated 
systems. The second law of thermodynamics states that the total entropy of an isolated system always increases or remains constant. 
In practical terms, this means that energy naturally disperses and becomes less organized. For instance, heat flows from hot objects 
to cold ones, not the other way around, because this increases overall entropy. Entropy helps explain why some processes are irreversible: 
they result in a net increase in disorder that cannot be undone without external work.""",
                
                # Google style: Informative, comprehensive, well-organized
                """Ionization is the process by which an atom or molecule gains or loses electrons, creating charged particles called ions. 
This can happen in several ways: through thermal energy when atoms are heated, through photons striking electrons, or through chemical reactions. 
When an electron is removed from an atom, it becomes a positively charged cation; when an electron is added, it becomes a negatively charged anion. 
Ionization is fundamental to many chemical processes, including the formation of ionic compounds, the behavior of gases in plasma, and the 
functioning of many biological processes in living organisms.""",
                
                # Google style: Educational, structured, informative
                """Refraction is the bending of light as it passes from one medium to another, occurring because light travels at different speeds 
in different materials. When light enters a denser medium like glass or water from air, it slows down and bends toward the normal line perpendicular 
to the surface. When it exits the denser medium back into air, it speeds up and bends away from the normal. This phenomenon is described by Snell's law 
and is responsible for optical effects like mirages, rainbows, and the apparent bending of objects seen through water or glass.""",
            ],
            
            "Anthropic": [
                # Anthropic style: Careful, includes nuance, somewhat cautious language
                """Exothermic reactions are chemical processes that release energy, typically in the form of heat, to their surroundings. 
When you observe an exothermic reaction, the surroundings typically become warmer. Common examples include combustion (burning) and many 
neutralization reactions. The released energy comes from the difference between the chemical bonds in the reactants and products—if the 
products have stronger bonds than the reactants, energy is released. It's worth noting that whether a reaction is exothermic or endothermic 
can depend on the specific conditions and the form in which the products are produced.""",
                
                # Anthropic style: Nuanced, acknowledges complexity, measured
                """Polymers are large molecules made up of many smaller units called monomers linked together in chains or networks. Natural 
polymers include proteins and DNA, while synthetic polymers include plastics and rubber. The properties of a polymer depend on the type of 
monomers used, how they're arranged, and the strength of the bonds between them. Different polymers have vastly different properties—some are 
flexible and stretchy, others are rigid and hard. It's important to note that the same monomer can produce polymers with different properties 
depending on how the monomers are connected and arranged.""",
                
                # Anthropic style: Thoughtful, includes caveats and nuance
                """Equilibrium in chemistry refers to a state where a reversible chemical reaction has reached a point where the forward and 
reverse reactions occur at equal rates. At equilibrium, the concentrations of reactants and products remain constant, though the reactions 
continue at the molecular level. The equilibrium position depends on various factors including temperature, pressure, and the presence of catalysts. 
It's worth noting that equilibrium is not the same as completion—a reaction may reach equilibrium long before all reactants are consumed, and the 
equilibrium point can shift if conditions change.""",
            ]
        }
    
    def classify(self, text: str) -> ClassificationResult:
        
        # Build the few-shot prompt
        prompt = self._build_prompt(text)
        
        # Call Gemini
        response = self.model.generate_content(prompt)
        
        # Tokens used
        tokens_used = 0
        try:
            tokens_used = response.usage_metadata.total_token_count
        except:
            pass
            
        # Parse the response
        response_text = response.text
        result = self._parse_response(
            text, 
            response_text, 
            tokens_used
        )
        
        return result
    
    def classify_batch(self, texts: list, show_progress: bool = True) -> list:
        results = []
        
        for i, text in enumerate(texts):
            result = self.classify(text)
            results.append(result)
            
            if show_progress and (i + 1) % 5 == 0:
                print(f"✓ Processed {i+1}/{len(texts)} texts")
        
        if show_progress:
            print(f"✓ Completed! Processed {len(texts)}/{len(texts)} texts")
        
        return results
    
    def get_results_summary(self, results: list) -> dict:
        classifications = {}
        confidence_by_class = {}
        total_tokens = 0
        
        for result in results:
            # Count classifications
            if result.classification not in classifications:
                classifications[result.classification] = 0
            classifications[result.classification] += 1
            
            # Track confidence by class
            if result.classification not in confidence_by_class:
                confidence_by_class[result.classification] = []
            confidence_by_class[result.classification].append(result.confidence)
            
            total_tokens += result.tokens_used
        
        # Calculate average confidence per class
        avg_confidence = {}
        for class_name, confidences in confidence_by_class.items():
            avg_confidence[class_name] = sum(confidences) / len(confidences)
        
        return {
            "total_texts": len(results),
            "classifications": classifications,
            "average_confidence_by_class": avg_confidence,
            "total_tokens_used": total_tokens,
            "estimated_cost_usd": (total_tokens / 1000) * 0.0001  # Rough estimate for Gemini Flash
        }
    
    def _build_prompt(self, text: str) -> str:
        
        prompt = """You are an expert at detecting whether scientific text (80-120 words) was written by a human or generated by an AI model. If it's AI-generated, you can identify which model generated it based on writing style and patterns.

STUDY THESE EXAMPLES CAREFULLY:

===== HUMAN-WRITTEN EXAMPLES =====
"""
        
        for i, example in enumerate(self.examples["Human"], 1):
            prompt += f"\nHuman Example {i}:\n{example}\n"
        
        prompt += "\n===== OPENAI-GENERATED EXAMPLES =====\n"
        for i, example in enumerate(self.examples["OpenAI"], 1):
            prompt += f"\nOpenAI Example {i}:\n{example}\n"
        
        prompt += "\n===== META-GENERATED EXAMPLES =====\n"
        for i, example in enumerate(self.examples["Meta"], 1):
            prompt += f"\nMeta Example {i}:\n{example}\n"
        
        prompt += "\n===== GOOGLE-GENERATED EXAMPLES =====\n"
        for i, example in enumerate(self.examples["Google"], 1):
            prompt += f"\nGoogle Example {i}:\n{example}\n"
        
        prompt += "\n===== ANTHROPIC-GENERATED EXAMPLES =====\n"
        for i, example in enumerate(self.examples["Anthropic"], 1):
            prompt += f"\nAnthropic Example {i}:\n{example}\n"
        
        prompt += f"""
===== TEXT TO CLASSIFY =====
{text}

===== YOUR TASK =====
Based on the examples above, classify the given text as one of these five categories:
- Human (human-written)
- OpenAI (written by OpenAI's models like ChatGPT/GPT-4)
- Meta (written by Meta's models like Llama)
- Google (written by Google's models like Gemini/Bard)
- Anthropic (written by Anthropic's models like Claude)

Key characteristics to look for:
- HUMAN: More varied structure, hedging language, informal tone, less consistent terminology
- OPENAI: Clear, confident, structured, somewhat formal, pedagogical
- META: Accessible, includes practical examples, balanced tone, informative
- GOOGLE: Well-structured, slightly formal, comprehensive, informative
- ANTHROPIC: Careful tone, includes nuance, measured language, acknowledges complexity

Respond with EXACTLY this format (no other text):
CLASSIFICATION: [Human/OpenAI/Meta/Google/Anthropic]
CONFIDENCE: [0-100]
REASONING: [One sentence explaining the key indicators]"""
        
        return prompt
    
    def _parse_response(self, text: str, response: str, tokens_used: int) -> ClassificationResult:
        """Parse the LLM response into structured format"""
        try:
            lines = response.strip().split('\n')
            classification = ""
            confidence = 0
            reasoning = ""
            
            for line in lines:
                if 'CLASSIFICATION:' in line:
                    classification = line.split(':', 1)[1].strip()
                elif 'CONFIDENCE:' in line:
                    conf_str = line.split(':', 1)[1].strip().replace('%', '')
                    try:
                        confidence = float(conf_str)
                    except:
                        confidence = 50
                elif 'REASONING:' in line:
                    reasoning = line.split(':', 1)[1].strip()
            
            # Validate classification
            valid_classes = ["Human", "OpenAI", "Meta", "Google", "Anthropic"]
            if classification not in valid_classes:
                print(f"Warning: Unexpected classification '{classification}', setting to 'Unknown'")
                classification = "Unknown"
            
            return ClassificationResult(
                text=text,
                classification=classification,
                confidence=confidence,
                reasoning=reasoning,
                tokens_used=tokens_used
            )
        except Exception as e:
            print(f"Warning: Could not parse response: {e}")
            return ClassificationResult(
                text=text,
                classification="Unknown",
                confidence=0,
                reasoning=f"Parse error: {str(e)}",
                tokens_used=tokens_used
            )
    
    def save_results_to_json(self, results: list, filepath: str):
        """Save classification results to JSON file"""
        json_data = []
        
        for result in results:
            json_data.append({
                "text": result.text[:100] + "..." if len(result.text) > 100 else result.text,
                "classification": result.classification,
                "confidence": result.confidence,
                "reasoning": result.reasoning,
                "full_text": result.text
            })
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Results saved to {filepath}")
    
    def save_results_to_csv(self, results: list, filepath: str):
        """Save classification results to CSV file"""
        import csv
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Text Preview', 'Classification', 'Confidence', 'Reasoning', 'Full Text'])
            
            for result in results:
                text_preview = result.text[:50] + "..." if len(result.text) > 50 else result.text
                writer.writerow([
                    text_preview,
                    result.classification,
                    result.confidence,
                    result.reasoning,
                    result.text
                ])
        
        print(f"✓ Results saved to {filepath}")

    def evaluate_on_dataset(self, csv_filepath: str) -> dict:
        import csv
        from collections import defaultdict
        
        texts = []
        true_labels = []
        
        with open(csv_filepath, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=';')
            header = next(reader, None)  # Skip header
            for row in reader:
                if len(row) >= 3:
                    texts.append(row[1])
                    true_labels.append(row[2])
                    
        print(f"Loaded {len(texts)} entries from {csv_filepath}")
        
        # Classify texts
        results = self.classify_batch(texts, show_progress=True)
        
        # Calculate accuracy
        correct = 0
        predictions = []
        
        for i, result in enumerate(results):
            pred = result.classification
            true_label = true_labels[i]
            predictions.append(pred)
            if pred == true_label:
                correct += 1
                
        accuracy = (correct / len(texts)) * 100 if texts else 0
        
        # Create detailed stats
        stats = defaultdict(lambda: {"total": 0, "correct": 0})
        for true_label, pred in zip(true_labels, predictions):
            stats[true_label]["total"] += 1
            if true_label == pred:
                stats[true_label]["correct"] += 1
                
        detailed_stats = {}
        for label, data in stats.items():
            detailed_stats[label] = {
                "accuracy": (data["correct"] / data["total"]) * 100 if data["total"] else 0,
                "total": data["total"]
            }
            
        return {
            "total_samples": len(texts),
            "correct_predictions": correct,
            "overall_accuracy": accuracy,
            "detailed_stats": detailed_stats,
            "results": results
        }


# ============================================================
# EXAMPLE USAGE
# ============================================================


def example_batch_texts():
    """Example 2: Classify multiple texts"""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Classify Multiple Texts")
    print("=" * 80)
    
    classifier = AISourceClassifier()
    
    # Multiple test texts from different sources
    test_texts = [
        """Osmosis is the movement of water molecules across a semipermeable membrane from areas where water is 
more concentrated to areas where it's less concentrated. This happens because water molecules are constantly moving 
around, and they tend to move toward regions where there's more dissolved stuff. This process is important in cells 
and is why putting salt on a wound can draw fluid out of nearby tissue.""",
        
        """Electrochemistry is the study of chemical reactions that produce electrical current or are caused by electrical current. 
In electrochemical cells, chemical energy is converted to electrical energy through oxidation-reduction reactions. Applications 
include batteries, fuel cells, and electroplating. The process involves the transfer of electrons between chemical species at different 
electrodes. Understanding electrochemistry is essential for energy storage and many industrial processes.""",
        
        """Polymer chemistry deals with the synthesis and properties of polymers, which are large molecules composed of repeating structural 
units. The monomers are linked together through covalent bonds to form long chains or networks. Different polymers exhibit vastly different 
properties depending on their monomer composition, chain structure, and intermolecular forces. Polymers are ubiquitous in modern materials, 
ranging from plastics and rubber to proteins and DNA.""",
    ]
    
    print(f"\nClassifying {len(test_texts)} texts...\n")
    results = classifier.classify_batch(test_texts)
    
    print("\n" + "-" * 80)
    print("RESULTS")
    print("-" * 80)
    
    for i, result in enumerate(results, 1):
        print(f"\nText {i}:")
        print(f"  Classification: {result.classification}")
        print(f"  Confidence: {result.confidence}%")
        print(f"  Reasoning: {result.reasoning}")
    
    # Summary
    print("\n" + "-" * 80)
    print("SUMMARY")
    print("-" * 80)
    summary = classifier.get_results_summary(results)
    print(f"Total texts: {summary['total_texts']}")
    print(f"Classifications: {summary['classifications']}")
    print(f"Average confidence by class:")
    for class_name, conf in summary['average_confidence_by_class'].items():
        print(f"  {class_name}: {conf:.1f}%")
    print(f"Total tokens used: {summary['total_tokens_used']}")
    print(f"Estimated cost: ${summary['estimated_cost_usd']:.3f}")


def example_save_results():
    """Example 3: Classify and save results"""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Classify and Save Results")
    print("=" * 80)
    
    classifier = AISourceClassifier()
    
    test_texts = [
        """Diffusion is the spontaneous movement of particles from high concentration to low concentration areas. 
It occurs due to random molecular motion and requires no energy input. Factors affecting diffusion rate include 
concentration gradient, temperature, molecular size, and medium viscosity. This fundamental process is critical in 
gas exchange, nutrient transport, and numerous chemical and biological systems.""",
        
        """Catalyst is a substance that increases the rate of a chemical reaction without being consumed in the process. 
It achieves this by lowering the activation energy required for the reaction to proceed. Catalysts are essential in 
industrial chemistry for improving efficiency and reducing energy consumption. Different catalysts work through various 
mechanisms, and the same catalyst can be used repeatedly for many reaction cycles.""",
    ]
    
    print(f"Classifying {len(test_texts)} texts...\n")
    results = classifier.classify_batch(test_texts, show_progress=True)
    
    # Save to JSON
    json_file = "/tmp/ai_source_results.json"
    classifier.save_results_to_json(results, json_file)
    
    # Save to CSV
    csv_file = "/tmp/ai_source_results.csv"
    classifier.save_results_to_csv(results, csv_file)
    
    print(f"\n✓ Results saved!")
    print(f"  JSON: {json_file}")
    print(f"  CSV: {csv_file}")

def example_evaluate_dataset():
    """Example 4: Evaluate accuracy on dataset_exemplos.csv"""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Evaluate on Dataset")
    print("=" * 80)
    
    import os
    dataset_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "dataset-exemplos.csv")
    dataset_path = os.path.abspath(dataset_path)
    
    if not os.path.exists(dataset_path):
        # Fallback to current dir relative data path
        dataset_path = os.path.join(os.getcwd(), "data", "dataset-exemplos.csv")
        
    if not os.path.exists(dataset_path):
        print(f"Could not find dataset at {dataset_path}")
        return
        
    classifier = AISourceClassifier()
    eval_results = classifier.evaluate_on_dataset(dataset_path)
    
    print("\n" + "-" * 80)
    print("EVALUATION RESULTS")
    print("-" * 80)
    print(f"Total samples tested: {eval_results['total_samples']}")
    print(f"Correct predictions:  {eval_results['correct_predictions']}")
    print(f"Overall Accuracy:     {eval_results['overall_accuracy']:.2f}%")
    
    print("\nAccuracy by class:")
    for label, stats in eval_results['detailed_stats'].items():
        print(f"  {label:<10} : {stats['accuracy']:.2f}% ({stats['total']} samples)")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    
    try:
        # Run examples
        # example_batch_texts()
        # example_save_results()  # Uncomment to save results
        example_evaluate_dataset()
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nTroubleshooting:")
        print("  1. Check GOOGLE_API_KEY is set:")
        print("     export GOOGLE_API_KEY='AIza...'")
        print("  2. Install Google Generative AI SDK:")
        print("     pip install google-generativeai")
        print("  3. Make sure you have internet connection")
        print("\nFull error:")
        import traceback
        traceback.print_exc()