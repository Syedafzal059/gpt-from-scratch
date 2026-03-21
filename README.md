# GPT From Scratch (WIP)

## Phase 1: Core Building Blocks
- Implemented tokenizer (char + word level)
- Built embedding layer
- Implemented scaled dot-product attention
- Built multi-head attention from scratch

## Phase 2: GPT Architecture ✅

- Implemented Transformer Block (Pre-Norm)
- Built Feed Forward Network (FFN)
- Stacked multiple transformer blocks
- Added token + positional embeddings
- Built full GPT model with output head

### Model Flow
Input → Embedding → Transformer Blocks → LayerNorm → Linear Head → Logits

## Phase 3: Training & Generation ✅

- Built dataset pipeline (next-token prediction)
- Implemented training loop (CrossEntropy + AdamW)
- Successfully trained model (loss reduced from ~3.0 to ~0.05)
- Implemented autoregressive text generation

### Sample Output
Input: "h"
Output: "hrshafdflasgffdzro..."

(Note: noisy output due to small dataset — expected behavior)

## Next Steps

## Goal
Understand and implement GPT architecture from first principles
