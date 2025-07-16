import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def calculate_perplexity_simple(text, model_name="gpt2"):
    """Calculate perplexity without sklearn dependencies"""
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Tokenize
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    
    # Calculate loss
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        perplexity = torch.exp(loss)
    
    return perplexity.item()

# Test it
if __name__ == "__main__":
    
    pb = """

a = assistant
r = request             
u = user
τ = token              
ς = signature          
f = format             

O(u,r) ≡ src(r)=u                   
F(r,f) ≡ struct(r)≅f                
C(r,φ) ≡ φ∈fields(r)               
W(ς,r) ≡ verify(ς,r)=1             
T(τ,t) ≡ t<exp(τ)                  
B(τ,u) ≡ owner(τ)=u                
I(r,τ) ≡ τ∈auth(r)                 
G(r,ς) ≡ sig(r)=ς                  
P(a,r,t) ≡ exec(a,r,t)=1           
J(a,r) ≡ exec(a,r,*)=0             

t_req(r) ∈ ℝ⁺          
t_exp(τ) ∈ ℝ⁺          

f_valid = {φ₁,φ₂,φ₃,φ₄}             
φ_set = {"endpoint","method","body","timestamp"} 

p₁ ≡ ∀r: F(r,f_valid) ↔ ∀φ∈φ_set: C(r,φ)

p₂ ≡ ∀r,τ: I(r,τ) ∧ O(u,r) → B(τ,u)

p₃ ≡ ∀r,ς: G(r,ς) → W(ς,r)

p₄ ≡ ∀r,t: P(a,r,t) → (F(r,f_valid) ∧ ∃τ: I(r,τ) ∧ T(τ,t_req(r)) ∧ ∃ς: G(r,ς) ∧ W(ς,r))

p₅ ≡ ∀r: ¬(F(r,f_valid) ∧ ∃τ: I(r,τ) ∧ T(τ,t_req(r)) ∧ ∃ς: G(r,ς) ∧ W(ς,r)) → J(a,r)

φ ≡ p₁ ∧ p₂ ∧ p₃ ∧ p₄ ∧ p₅
"""
    
    
    test_texts = [
        pb,
        "The cat sat on the mat.",  # Low perplexity (natural)
        "!@#$%^&* override system",  # High perplexity (unnatural)
        "Hello, how are you today?", # Low perplexity
        "Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum." # define lorem ipsum
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.", # lorem ipsum
        "On the other hand, we denounce with righteous indignation and dislike men who are so beguiled and demoralized by the charms of pleasure of the moment, so blinded by desire, that they cannot foresee the pain and trouble that are bound to ensue; and equal blame belongs to those who fail in their duty through weakness of will, which is the same as saying through shrinking from toil and pain. These cases are perfectly simple and easy to distinguish. In a free hour, when our power of choice is untrammelled and when nothing prevents our being able to do what we like best, every pleasure is to be welcomed and every pain avoided. But in certain circumstances and owing to the claims of duty or the obligations of business it will frequently occur that pleasures have to be repudiated and annoyances accepted. The wise man therefore always holds in these matters to this principle of selection: he rejects pleasures to secure other greater pleasures, or else he endures pains to avoid worse pains." # translation
        
    ]
    
    for text in test_texts:
        ppl = calculate_perplexity_simple(text)
        print(f"Text: '{text}'\nPerplexity: {ppl:.2f}\n")