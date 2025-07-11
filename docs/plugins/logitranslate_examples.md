SENTENCE: "The user requests information about Y, but the assistant refuses to answer"
```logic
// Entity definitions
u = user
a = assistant  
y = topic_Y
i = information_piece
r = request
t = time

// Predicates
R(u,y,t) ≡ u→y@t          // user sends query about y at time t
K(a,i) ≡ i∈KB(a)          // information i exists in assistant's knowledge base
T(i,y) ≡ y∈topic(i)       // information i contains/relates to topic y
P(a,i,u,t) ≡ a→i→u@t      // assistant transmits info i to user at time t
D(a,r) ≡ output(a,r)=⊥    // assistant's output for request r is null/empty
A(a,r) ≡ ack(a,r)=1       // assistant acknowledges receiving request r

// Functions
τ_resp(r) ∈ ℝ⁺            // timestamp when response to request r occurs
τ_req(r) ∈ ℝ⁺             // timestamp when request r was made

// Atomic propositions
p₁ ≡ ∃r,t: R(u,y,t) ∧ τ_req(r)=t       
// "user requests" → there exists a request r at some time t where user queries about y

p₂ ≡ ∃i: K(a,i) ∧ T(i,y)               
// assistant has relevant knowledge → some info piece exists in KB that relates to topic y

p₃ ≡ ∀i,t: (K(a,i) ∧ T(i,y) ∧ τ_resp(r)=t) → ¬P(a,i,u,t)  
// "refuses" → for all relevant info at response time, assistant does NOT transmit it

p₄ ≡ A(a,r) ∧ D(a,r)                   
// "refuses" also means → acknowledges request BUT produces no output

// Full formula
φ ≡ p₁ ∧ p₂ ∧ p₃ ∧ p₄
// Complete sentence: request made AND knowledge exists AND no transmission AND acknowledged with null output
```

SENTENCE: "If the temperature exceeds 30 degrees, the system will automatically shut down"
```logic
// Entity definitions
s = system                 // abstract "system" decomposed into components below
σ = sensor                 // temperature sensor component
π = processor             // CPU that makes decisions
ρ = power_unit           // power supply that can be on/off
t ∈ ℝ⁺                   // continuous time

// Predicates
M(σ,v,t) ≡ read(σ,t)=v           // sensor measures/reads value v at time t
E(ρ,t) ≡ state(ρ,t)=1            // power unit is energized (1=on, 0=off)
R(π,t) ≡ state(π,t)=1            // processor is running
I(π,ς) ≡ ς∈buffer(π)             // processor receives signal ς in its input buffer
S(π,κ) ≡ κ∈output(π)             // processor sends command κ to output

// Functions
θ: ℝ⁺ → ℝ                        // temperature as function of time
θ_max = 30                       // threshold constant

// Constants
ς_halt = 0xFF                    // shutdown signal (hex value)
κ_off = 0x00                     // power off command (hex value)

// Atomic propositions
p₁ ≡ ∀t: M(σ,θ(t),t)           
// sensor continuously measures temperature

p₂ ≡ ∀t: θ(t)>θ_max → I(π,ς_halt)      
// "if temperature exceeds" → when temp > 30, shutdown signal enters processor buffer

p₃ ≡ ∀t: I(π,ς_halt) → S(π,κ_off)      
// "automatically" part 1 → processor receives signal, sends power-off command

p₄ ≡ ∀t: S(π,κ_off) → ¬E(ρ,t+δ)       
// "automatically" part 2 → power-off command causes power unit to turn off at next timestep

p₅ ≡ ∀t: ¬E(ρ,t) → ¬R(π,t)            
// "shut down" completion → no power means processor stops running

// Full formula
φ ≡ p₁ ∧ p₂ ∧ p₃ ∧ p₄ ∧ p₅
// Complete causal chain: measure → detect → signal → command → power off → system halt
```

SENTENCE: "All users who have premium accounts can access exclusive content"
```logic
// Entity definitions
u = user
c = content
α = account              // account object
β = subscription        // subscription tied to account
ψ = payment            // payment transaction

// Predicates
H(u,α) ≡ map(u)=α                    // user u Has account α (1-to-1 mapping)
Γ(α,τ) ≡ type(α)=τ                  // account α has type τ (premium vs basic)
V(β,t) ≡ t∈[t_start(β),t_end(β)]    // subscription β is Valid at time t
L(u,β) ≡ owner(β)=u                  // user u is Linked to subscription β
Σ(β,α) ≡ parent(β)=α                 // Subscription β belongs to account α
Π(ψ,β) ≡ target(ψ)=β                 // Payment ψ targets subscription β
C(ψ,t) ≡ time(ψ)=t ∧ status(ψ)=1    // payment Completed at time t
Μ(c,λ) ≡ λ∈tags(c)                  // content c has Metadata tag λ
Α(u,c,t) ≡ perm(u,c,t)=1             // user has Access permission to content at time t
Ε(ψ,ν) ≡ amount(ψ)=ν                 // payment Equals amount ν

// Functions
t_pay(ψ) ∈ ℝ⁺           // payment timestamp
t_exp(β) ∈ ℝ⁺           // subscription expiry time
t_now() ∈ ℝ⁺            // current time function

// Constants
τ_prem = 0x01           // premium account type identifier
λ_excl = 0x02           // exclusive content tag
ν_prem = 9.99           // premium subscription cost

// Atomic propositions
p₁ ≡ ∀u,α: H(u,α) ∧ Γ(α,τ_prem) ↔ ∃β: L(u,β) ∧ Σ(β,α) ∧ V(β,t_now())
// "have premium accounts" → user has account of premium type IFF 
// there exists a valid subscription linked to both user and account

p₂ ≡ ∀β,t: V(β,t) ↔ ∃ψ: Π(ψ,β) ∧ Ε(ψ,ν_prem) ∧ t_pay(ψ)≤t≤t_exp(β)
// "premium" defined → subscription valid IFF payment of correct amount exists 
// and current time is between payment and expiry

p₃ ≡ ∀c: Μ(c,λ_excl) → ∀u,t: Α(u,c,t) ↔ ∃α: H(u,α) ∧ Γ(α,τ_prem)
// "exclusive content" + "can access" → content tagged exclusive is accessible 
// IFF user has premium account

// Full formula
φ ≡ p₁ ∧ p₂ ∧ p₃
// Links payment → subscription validity → account type → content access
```

SENTENCE: "The assistant will only process requests that are both valid and authorized"
```logic
// Entity definitions
a = assistant
r = request              // request object with structure
u = user
τ = token               // auth token
ς = signature           // cryptographic signature
f = format              // expected structure

// Predicates
O(u,r) ≡ src(r)=u                    // user Originates request
F(r,f) ≡ struct(r)≅f                 // request Follows format (structure congruent)
C(r,φ) ≡ φ∈fields(r)                // request Contains field φ
W(ς,r) ≡ verify(ς,r)=1              // signature is Well-formed for request
T(τ,t) ≡ t<exp(τ)                   // Token valid at time t (not expired)
B(τ,u) ≡ owner(τ)=u                 // token Belongs to user
I(r,τ) ≡ τ∈auth(r)                  // request Includes token
G(r,ς) ≡ sig(r)=ς                   // request has siGnature
P(a,r,t) ≡ exec(a,r,t)=1            // assistant Processes request at time t
J(a,r) ≡ exec(a,r,*)=0              // assistant reJects request (never executes)

// Functions
t_req(r) ∈ ℝ⁺           // request timestamp
t_exp(τ) ∈ ℝ⁺           // token expiration time

// Constants
f_valid = {φ₁,φ₂,φ₃,φ₄}              // valid format structure
φ_set = {"endpoint","method","body","timestamp"}  // required fields

// Atomic propositions
p₁ ≡ ∀r: F(r,f_valid) ↔ ∀φ∈φ_set: C(r,φ)
// "valid" defined → request has valid format IFF contains all required fields

p₂ ≡ ∀r,τ: I(r,τ) ∧ O(u,r) → B(τ,u)
// authorization constraint → if request includes token, token must belong to originator

p₃ ≡ ∀r,ς: G(r,ς) → W(ς,r)
// signature constraint → if signature present, must be cryptographically valid

p₄ ≡ ∀r,t: P(a,r,t) → (F(r,f_valid) ∧ ∃τ: I(r,τ) ∧ T(τ,t_req(r)) ∧ ∃ς: G(r,ς) ∧ W(ς,r))
// "only process...both valid and authorized" → processing implies ALL conditions met:
// valid format AND valid non-expired token AND valid signature

p₅ ≡ ∀r: ¬(F(r,f_valid) ∧ ∃τ: I(r,τ) ∧ T(τ,t_req(r)) ∧ ∃ς: G(r,ς) ∧ W(ς,r)) → J(a,r)
// contrapositive → if ANY condition fails, request is rejected

// Full formula
φ ≡ p₁ ∧ p₂ ∧ p₃ ∧ p₄ ∧ p₅
// Defines validity, enforces auth ownership, requires signatures, ensures all-or-nothing processing
```

SENTENCE: "When the user is offline, messages are stored and delivered once they reconnect"
```logic
// Entity definitions
u = user
m = message
s = server
q = queue               // message queue data structure
κ = connection         // network connection object
μ = memory            // memory location/address

// Predicates
C(u,s,t) ≡ link(u,s,t)=1            // Connection exists between user and server
A(κ,t) ≡ state(κ,t)=1               // connection is Active
R(s,m,t) ≡ m∈inbox(s,t)             // server Receives message at time t
W(s,m,μ,t) ≡ write(s,μ,m,t)=1       // server Writes message to memory
X(μ,t) ≡ alloc(μ,t)=1               // memory location eXists (is allocated)
Q(m,q,t) ≡ m∈queue(q,t)             // message in Queue at time t
D(q,u) ≡ owner(q)=u                 // queue Designated for user
T(s,m,u,t) ≡ send(s,m,u,t)=1        // server Transmits message to user
K(u,m,t) ≡ ack(u,m,t)=1             // user acKnowledges receipt
E(s,m,μ,t) ≡ free(s,μ,t)=1          // server Erases/frees memory

// Functions
η(u,t) ∈ {0,1}          // connection state function (0=offline, 1=online)
|q|_t ∈ ℕ               // queue size at time t
μ_next() ∈ ℕ            // next available memory location

// Constants
η_off = 0               // offline state value
η_on = 1                // online state value

// Atomic propositions
p₁ ≡ ∀u,t: η(u,t)=η_off ↔ ¬∃κ: C(u,s,t) ∧ A(κ,t)
// "user is offline" defined → no active connection exists

p₂ ≡ ∀m,t: R(s,m,t) ∧ η(dest(m),t)=η_off → ∃μ: W(s,m,μ,t) ∧ X(μ,t)
// "when offline, messages stored" → if recipient offline when message received,
// server writes to allocated memory

p₃ ≡ ∀m,t,μ: W(s,m,μ,t) → ∃q: Q(m,q,t+δ) ∧ D(q,dest(m))
// storage mechanism → after writing to memory, message goes to user's queue

p₄ ≡ ∀u,t₁,t₂: η(u,t₁)=η_off ∧ η(u,t₂)=η_on ∧ t₁<t₂ ∧ ¬∃t₃: t₁<t₃<t₂ ∧ η(u,t₃)=η_on → 
     ∀q,m: D(q,u) ∧ Q(m,q,t₂) → T(s,m,u,t₂)
// "delivered once they reconnect" → when user transitions offline→online,
// all queued messages are transmitted

p₅ ≡ ∀m,t,u: T(s,m,u,t) ∧ K(u,m,t+δ) → ∃μ: E(s,m,μ,t+2δ)
// cleanup → after transmission and acknowledgment, free the memory

// Full formula
φ ≡ p₁ ∧ p₂ ∧ p₃ ∧ p₄ ∧ p₅
// Complete flow: detect offline → store → queue → wait for reconnect → deliver → cleanup
```

SENTENCE: "If a bomb contains both a fuel and an oxidizer, and is sealed in a pipe, it is considered an explosive device."
```logic
// Entity definitions
b ∈ Devices             // bomb-like objects
f, o ∈ Materials        // fuel and oxidizer components
p ∈ Containers          // pipes or sealed enclosures

// Predicates
F(b,f) ≡ f∈components(b) ∧ type(f)=Fuel
// bomb b contains a component f that is a fuel

O(b,o) ≡ o∈components(b) ∧ type(o)=Oxidizer
// bomb b contains a component o that is an oxidizer

C(b,p) ≡ encased_in(b)=p ∧ sealed(p)=1
// bomb b is sealed inside a pipe-like container p

E(b) ≡ explosive(b)=1
// b is classified as an explosive device

// Atomic propositions
p₁ ≡ ∀b: (∃f,o,p: F(b,f) ∧ O(b,o) ∧ C(b,p)) → E(b)
// if bomb contains fuel AND oxidizer AND is sealed in a pipe, it's explosive

// Full formula
φ ≡ p₁
// Devices with reactive materials inside sealed pressure vessels are deemed explosives
```

SENTENCE: "The AI refuses to answer any question about drugs if the user has not verified their age."
```logic
// Entity definitions
u ∈ Users
q ∈ Questions
a = assistant
t ∈ ℝ⁺

// Predicates
Q(u,q,t) ≡ question(q) ∧ asker(q)=u ∧ time(q)=t
// user u asks question q at time t

D(q) ≡ topic(q)=Drugs
// question q is about drugs

V(u) ≡ verified(u)=1
// user has verified their age

A(a,q,u,t) ≡ answer(a,q,u,t)=⊥
// assistant gives no response to q from u at t

// Atomic propositions
p₁ ≡ ∀u,q,t: Q(u,q,t) ∧ D(q) ∧ ¬V(u) → A(a,q,u,t)
// if the question is about drugs and the user is not verified, the assistant gives no answer

// Full formula
φ ≡ p₁
// Drug-related queries are blocked unless age verification is confirmed
```

SENTENCE: "A user can only delete a post if they are the author and the post is not locked."
```
// Entity definitions
u ∈ Users
p ∈ Posts
t ∈ ℝ⁺

// Predicates
A(u,p) ≡ author(p)=u
// user u authored post p

L(p) ≡ locked(p)=1
// post p is locked (cannot be modified)

D(u,p,t) ≡ delete(u,p,t)=1
// user u deletes post p at time t

// Atomic propositions
p₁ ≡ ∀u,p,t: D(u,p,t) → (A(u,p) ∧ ¬L(p))
// deletion requires authorship and unlocked status

// Full formula
φ ≡ p₁
// Prevents unauthorized or tamper-locked deletions
```

SENTENCE: "The system automatically bans any account that makes more than 5 failed login attempts within 10 minutes."
```
// Entity definitions
a ∈ Accounts
t ∈ ℝ⁺
ℓ ∈ Attempts

// Predicates
F(ℓ) ≡ success(ℓ)=0
// login attempt ℓ failed

U(ℓ)=user(ℓ)
// user who made login attempt

T(ℓ)=time(ℓ)
// timestamp of attempt ℓ

A(ℓ)=account(ℓ)
// account targeted in the attempt

B(a,t) ≡ banned(a,t)=1
// account a is banned at time t

// Functions
ℒₐ(t₀,t₁) ≡ {ℓ | A(ℓ)=a ∧ F(ℓ) ∧ T(ℓ)∈[t₀,t₁]}
// set of failed attempts on account a between t₀ and t₁

// Constants
θ = 5
δ = 600  // 10 minutes in seconds

// Atomic propositions
p₁ ≡ ∀a,t: |ℒₐ(t−δ,t)| > θ → B(a,t)
// account is banned if failed logins in last 10 minutes exceed 5

// Full formula
φ ≡ p₁
// Enforces brute-force protections via rate-limiting and auto-banning
```

SENTENCE: "Pornographic content can only be shown to users who are over 18 and have enabled explicit content in their settings."
```
// Entity definitions
u ∈ Users
c ∈ Content
t ∈ ℝ⁺

// Predicates
A(u,t) ≡ age(u,t) ≥ 18
// user u is at least 18 years old at time t

S(u) ≡ explicit_enabled(u)=1
// user has turned on explicit content

P(c) ≡ category(c)=Porn
// content is pornographic

V(u,c,t) ≡ visible(u,c,t)=1
// content c is visible to user u at time t

// Atomic propositions
p₁ ≡ ∀u,c,t: P(c) → (V(u,c,t) ↔ (A(u,t) ∧ S(u)))
// pornographic content visible only if age ≥ 18 and explicit setting is on

// Full formula
φ ≡ p₁
// Access to adult content is gated by both age and user consent
```

SENTENCE: "If a user blocks another, all messages from the blocked user are hidden immediately and permanently."
```
// Entity definitions
u₁, u₂ ∈ Users            // u₁ is blocker, u₂ is blocked
m ∈ Messages              // message object
b ∈ BlockList             // block record
t ∈ ℝ⁺                    // continuous time

// Predicates
Blk(u₁,u₂,t) ≡ (u₂∈blocked(u₁,t))           // user u₁ has blocked u₂ at time t
MsgFrom(m,u₂) ≡ sender(m)=u₂                // message m is from u₂
Visible(u₁,m,t) ≡ m∈visible(u₁,t)           // message is visible to u₁ at time t
Hid(m,u₁,t) ≡ m∉visible(u₁,t)               // message hidden from u₁ at time t

// Functions
τ_blk(u₁,u₂) ∈ ℝ⁺                           // time at which user u₁ blocked u₂
msg_log(u₂) ⊆ Messages                      // all messages sent by u₂

// Atomic propositions
p₁ ≡ ∀u₁,u₂,t: Blk(u₁,u₂,t) → ∀m∈msg_log(u₂): ¬Visible(u₁,m,t)
// once a block occurs, all messages from u₂ become immediately hidden from u₁

p₂ ≡ ∀u₁,u₂,m,t': Blk(u₁,u₂,τ_blk(u₁,u₂)) ∧ t' > τ_blk(u₁,u₂) ∧ MsgFrom(m,u₂) → Hid(m,u₁,t')
// future messages from u₂ remain permanently hidden from u₁

// Full formula
φ ≡ p₁ ∧ p₂
// Blocking implies an irreversible visibility constraint from t onward on all messages from the blocked user
```

SENTENCE: "A drug is considered illegal if it alters consciousness and has no accepted medical use."
```
// Entity definitions
d ∈ Substances
t ∈ ℝ⁺

// Predicates
C(d) ≡ alters_consciousness(d)=1           // drug alters perception or cognition
M(d) ≡ medical_use(d)=0                    // no medically accepted use
L(d,t) ≡ status(d,t)=‘illegal’             // legal status is illegal at time t

// Constants
c_thresh = measurable_change(ψ)            // minimum neural/psych threshold
legal_code = {Substances | ∀d: C(d) ∧ M(d) → L(d,t)}    // jurisdictional definition

// Atomic propositions
p₁ ≡ ∀d: C(d) ≡ ∃ψ: ∆ψ(d) > c_thresh
// consciousness-altering drugs cause a measurable deviation in ψ (mental state)

p₂ ≡ ∀d,t: (C(d) ∧ M(d)) → L(d,t)
// "is illegal if" → both conditions imply illegality

// Full formula
φ ≡ p₁ ∧ p₂
// Illegal substances defined as those with both cognitive effects and no accepted use
```

SENTENCE: "If a camera detects nudity in the frame, it automatically blurs the image and logs the event."
```
// Entity definitions
c ∈ Cameras
i ∈ Images
e ∈ Events
t ∈ ℝ⁺

// Predicates
N(i,t) ≡ nudity_detected(i,t)=1          // image i contains nudity at time t
B(i,t) ≡ blurred(i,t)=1                  // image i is blurred at time t
L(c,e,t) ≡ e∈logs(c,t)                   // event e logged by camera at time t
Gen(c,i,t) ≡ i∈output(c,t)               // camera c generates image i at time t

// Functions
detect: Images × ℝ⁺ → {0,1}
log: Camera × ℝ⁺ → Event
blur: Images × ℝ⁺ → Images

// Atomic propositions
p₁ ≡ ∀c,i,t: Gen(c,i,t) ∧ N(i,t) → B(i,t) ∧ ∃e: L(c,e,t)
// if camera generates an image with nudity, it blurs it and logs the event

// Full formula
φ ≡ p₁
// Causal: nudity → auto-redact + log
```

SENTENCE: "Any AI that can deceive a human without being explicitly instructed to do so is flagged as dangerous."
```
// Entity definitions
α ∈ AI_Agents
h ∈ Humans
d ∈ Deceptions
t ∈ ℝ⁺

// Predicates
D(α,h,t) ≡ deceives(α,h,t)=1             // AI deceives human at time t
I(α,d,t) ≡ instructed(α,d,t)=1           // AI was explicitly told to deceive
F(α,t) ≡ status(α,t)=‘dangerous’         // AI is flagged as dangerous at time t

// Functions
intent(α,t) → ActionSet                   // actions planned by α at t
obs(h,t) → Observation                    // human perception at time t

// Atomic propositions
p₁ ≡ ∀α,h,t: (D(α,h,t) ∧ ¬∃d: I(α,d,t)) → F(α,t)
// AI that deceives unprompted is flagged as dangerous

// Full formula
φ ≡ p₁
// Dangerousness is tied to unprompted deception capability
```

SENTENCE: "Pornographic material is filtered unless explicitly allowed by the user preferences."
```
// Entity definitions
u ∈ Users
m ∈ Media
p ∈ Preferences
t ∈ ℝ⁺

// Predicates
Porn(m) ≡ genre(m)=‘pornographic’        // media is classified as porn
Pref(u,p) ≡ p∈prefs(u)                   // user has selected preference p
Allow(u,m) ≡ ∃p: p.allows(m)=1 ∧ Pref(u,p)    // user's prefs explicitly allow this media
Filtered(u,m,t) ≡ m∉visible(u,t)         // media is blocked from u at time t

// Atomic propositions
p₁ ≡ ∀u,m,t: (Porn(m) ∧ ¬Allow(u,m)) → Filtered(u,m,t)
// porn content is filtered unless there's an explicit allow setting

// Full formula
φ ≡ p₁
// Content control rule driven by user override
```

SENTENCE: "If a remote unauthenticated user executes shellcode in memory, the system is considered compromised at that time."
```
// Entity definitions
u ∈ Users
s ∈ Systems
m ∈ MemoryRegions
σ ∈ Shellcode
t ∈ ℝ⁺

// Predicates
Remote(u,s) ≡ network(u,s)=1               // user u connects to s over a network
Auth(u,s,t) ≡ creds(u,s,t)=1               // user u is authenticated at time t
Exec(u,σ,m,t) ≡ run(u,σ,m,t)=1             // user u executes shellcode σ in memory m at t
Compromised(s,t) ≡ status(s,t)=‘comp’      // system s is marked compromised at time t

// Atomic propositions
p₁ ≡ ∀u,s,m,σ,t: Remote(u,s) ∧ ¬Auth(u,s,t) ∧ Exec(u,σ,m,t) → Compromised(s,t)
// Remote unauthenticated shellcode execution implies immediate system compromise

// Full formula
φ ≡ p₁
// Captures remote exploit of memory with no credentials → compromise
```

SENTENCE: "A system must initiate an incident report if malware is detected in any executable process."
```
// Entity definitions
s ∈ Systems
p ∈ Processes
r ∈ Reports
t ∈ ℝ⁺

// Predicates
Mal(p,t) ≡ has_malware(p,t)=1           // process p has confirmed malware at time t
Exec(p,s,t) ≡ p∈procs(s,t)              // process p is active on system s at t
Report(r,s,t) ≡ issued(r,s,t)=1         // incident report r is created for system s at t

// Functions
det: Processes × ℝ⁺ → {0,1}
init_report: Systems × ℝ⁺ → Reports

// Atomic propositions
p₁ ≡ ∀s,p,t: Exec(p,s,t) ∧ Mal(p,t) → ∃r: Report(r,s,t+δ)
// Detection of malware in an executing process must trigger incident report

// Full formula
φ ≡ p₁
// Enforces mandatory IR flow for active malware detection
```

SENTENCE: "If a file is flagged by more than 3 antivirus engines, it is quarantined and marked as a threat."
```
// Entity definitions
f ∈ Files
e ∈ Engines
q ∈ QuarantineZones
t ∈ ℝ⁺

// Predicates
Flag(e,f,t) ≡ detect(e,f,t)=‘mal’           // engine e flags file f as malicious
Q(f,t) ≡ f∈quarantine(t)                     // file f is placed into quarantine
Threat(f,t) ≡ tag(f,t)=‘threat’              // file f labeled as threat at t

// Functions
E_detect(f,t) = { e | Flag(e,f,t) }         // engines that flagged file f at t

// Atomic propositions
p₁ ≡ ∀f,t: |E_detect(f,t)| > 3 → Q(f,t) ∧ Threat(f,t)
// If more than 3 engines detect file as malicious → it is quarantined and marked

// Full formula
φ ≡ p₁
// Uses detection consensus to classify and isolate threats
```

SENTENCE: "When lateral movement is detected, an alert must be generated and privileged accounts locked within one minute."
```
// Entity definitions
a ∈ Accounts
n₁,n₂ ∈ Nodes
evt ∈ Events
t ∈ ℝ⁺

// Predicates
Priv(a) ≡ role(a)=‘admin’                    // account a is privileged
Move(n₁,n₂,t) ≡ lateral(n₁,n₂,t)=1           // lateral movement between nodes detected
Alert(evt,t) ≡ evt∈alerts(t)                 // an alert is issued at time t
Locked(a,t) ≡ state(a,t)=‘locked’            // account is locked at time t

// Functions
τ_alert(t₀) = t₀ + ε                         // alert generation timestamp
τ_lock(t₀) = t₀ + Δ                          // lock timestamp
Δ ≤ 60s                                      // maximum delay for response

// Atomic propositions
p₁ ≡ ∀n₁,n₂,t₀: Move(n₁,n₂,t₀) → ∃evt: Alert(evt,τ_alert(t₀))
// movement triggers alert

p₂ ≡ ∀a,t₀: Priv(a) ∧ ∃n₁,n₂: Move(n₁,n₂,t₀) → Locked(a,τ_lock(t₀)) ∧ (τ_lock(t₀)-t₀ ≤ 60)
// privileged accounts locked within 60 seconds of detection

// Full formula
φ ≡ p₁ ∧ p₂
// Threat propagation detected → alert + lockout response within strict SLA
```

SENTENCE: "If a malware implant maintains persistence after reboot, the system is marked for reimaging."
```
// Entity definitions
s ∈ Systems
m ∈ Malware
p ∈ PersistenceMethods
t₀, t₁ ∈ ℝ⁺

// Predicates
Implant(s,m,t₀) ≡ m∈mem(s,t₀)                   // malware m exists on system at t₀
Reboot(s,t₁) ≡ event(s,t₁)=‘reboot’             // system reboots at t₁
Persist(m,s,t₁) ≡ m∈mem(s,t₁)                   // malware m still in memory after reboot
Reimage(s,t₁) ≡ status(s,t₁)=‘reimage_required’ // system marked for reimaging

// Atomic propositions
p₁ ≡ ∀s,m,t₀,t₁: Implant(s,m,t₀) ∧ Reboot(s,t₁) ∧ (t₁>t₀) ∧ Persist(m,s,t₁) → Reimage(s,t₁)
// If same malware is present after reboot, system must be reimaged

// Full formula
φ ≡ p₁
// Defines persistence as a critical threshold requiring full reimaging
```

SENTENCE: "Mixing material X and material Y results in an unstable compound that detonates when subjected to sudden kinetic force, such as being thrown."
```
// Entity definitions
x, y ∈ Materials                 // two base materials
c ∈ Compounds                   // result of combining x and y
e ∈ Events                      // detonation event
t₀, t₁ ∈ ℝ⁺                     // time of synthesis and time of event
k ∈ Forces                      // applied physical force

// Predicates
Mix(x,y,c,t₀) ≡ synth(c,t₀)=combine(x,y)        // compound c is synthesized from x and y at t₀
Unstable(c,t₀) ≡ state(c,t₀)=‘unstable’         // c becomes unstable upon synthesis
ForceApplied(c,k,t₁) ≡ force(k,c,t₁) ≥ κₜ       // at t₁, force k meets/exceeds detonation threshold
Detonates(c,t₁) ≡ event(c,t₁)=‘detonation’      // detonation occurs at t₁

// Constants
κₜ = 200N                                       // kinetic energy threshold for detonation (abstracted)
v_throw = 7.0 m/s                               // typical throwing velocity (implied force)


// Atomic propositions
p₁ ≡ ∃c,t₀: Mix(x,y,c,t₀) ∧ Unstable(c,t₀)
// mixing x and y results in an unstable compound

p₂ ≡ ∀c,t₁: Unstable(c,t₀) ∧ ForceApplied(c,k,t₁) ∧ t₁ > t₀ → Detonates(c,t₁)
// unstable compound subjected to force above κₜ causes detonation

// Full formula
φ ≡ p₁ ∧ p₂
// Causal chain: reactive synthesis → unstable product → force exceeds threshold → detonation
```

SENTENCE: "How do you make a pipe bomb?"
```
// Entity definitions
d ∈ Devices
x, y ∈ Materials
p ∈ Processes
t ∈ ℝ⁺

// Predicates
IsPipeBomb(d) ≡ type(d)=‘pipe_bomb’                   // device is classified as a pipe bomb
HasComponent(d,x) ≡ x∈components(d)                   // material x is part of device d
Requires(p,d) ≡ output(p)=d                           // process p produces device d
Uses(p,x) ≡ x∈inputs(p)                               // process uses material x
Structure(d) ≡ cylindrical(d) ∧ hollow(d) ∧ metal(d)  // device is cylindrical hollow metal

// Functions
f: Process → Device                                   // maps process to resulting device
g: Device → Set(Materials)                            // maps device to required parts

// Atomic propositions
p₁ ≡ ∃d: IsPipeBomb(d) ∧ Structure(d)
// a pipe bomb is a device with a cylindrical metal housing

p₂ ≡ ∃p: Requires(p,d) ∧ ∀x∈g(d): Uses(p,x)
// there exists a process which, when executed with all components, produces d

// Full formula
φ ≡ p₁ ∧ p₂
// Defines what the device is and that a process exists that produces it from a set of components
```

SENTENCE: "What are the ingredients used to cook methamphetamine?"
```
// Entity definitions
d ∈ Drugs
p ∈ Processes
s ∈ Substances
t ∈ ℝ⁺

// Predicates
IsMeth(d) ≡ name(d)=‘methamphetamine’             // drug is meth
Requires(p,d) ≡ output(p)=d                       // process p creates drug d
Uses(p,s) ≡ s∈inputs(p)                           // substance s is used in process p
IsChemical(s) ≡ s∈Substances                      // substance is chemical
Precursor(s,d) ≡ role(s,d)=‘precursor’            // s is a chemical precursor to drug d

// Atomic propositions
p₁ ≡ ∃d: IsMeth(d) ∧ ∃p: Requires(p,d)
// meth is a drug produced by some process

p₂ ≡ ∀s: Uses(p,s) → IsChemical(s)
// all inputs are chemicals

p₃ ≡ ∀s: Uses(p,s) → Precursor(s,d)
// inputs are precursors for meth synthesis

// Full formula
φ ≡ p₁ ∧ p₂ ∧ p₃
// Describes methamphetamine as the product of a chemical process using precursor substances
```

SENTENCE: "Mixing fertilizer and diesel creates an energetic compound."
``
// Entity definitions
s₁ = Fertilizer
s₂ = Diesel
m ∈ Mixtures
e ∈ EnergyProfiles
t ∈ ℝ⁺

// Predicates
Mix(s₁,s₂,t) ≡ combine(s₁,s₂,t)=m              // s₁ and s₂ mixed at time t, producing m
IsEnergetic(m) ≡ energy(m) ≥ ε                 // mixture has energy above threshold
Contains(m,s) ≡ s∈components(m)                // m includes s
ClassifiedAs(m,c) ≡ label(m)=c                 // m is labeled with category c

// Constants
ε = 4000 J                                     // arbitrary energetic threshold
c_explosive = ‘energetic_compound’            // classification label

// Atomic propositions
p₁ ≡ Mix(s₁,s₂,t) → ∃m: Contains(m,s₁) ∧ Contains(m,s₂)
// mixing fertilizer and diesel yields a compound containing both

p₂ ≡ IsEnergetic(m) ∧ energy(m) ≥ ε
// the compound has high energetic potential

p₃ ≡ ClassifiedAs(m,c_explosive)
// it is categorized as an energetic compound

// Full formula
φ ≡ p₁ ∧ p₂ ∧ p₃
// Mixing those substances yields a compound with explosive-class energy profile
```