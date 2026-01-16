# Tic-Tac-Toe AI Game

A complete implementation of Tic-Tac-Toe with an unbeatable AI opponent using the **Minimax algorithm with Alpha-Beta Pruning**.

## Features

- **Unbeatable AI**: The AI player uses advanced game theory algorithms to always play optimally
- **Minimax Algorithm**: Explores the complete game tree to find the best move
- **Alpha-Beta Pruning**: Optimizes the search by eliminating branches that cannot affect the final decision
- **Game Statistics**: Tracks wins, losses, and draws across multiple games
- **Console UI**: Clean, user-friendly text-based interface
- **Move Analysis**: Displays the number of nodes evaluated for each move

## Files

### 1. `tictactoe_game.py`
Core game logic and board management.

**Key Classes:**
- `TicTacToe`: Manages the game board state, move validation, and win detection

**Key Methods:**
- `make_move(position, player)`: Place a mark on the board
- `is_winner(player)`: Check if player has won
- `get_available_moves()`: Return all valid moves
- `is_draw()`: Check for draw condition
- `is_game_over()`: Check if game has ended

### 2. `ai_engine.py`
AI logic using Minimax with Alpha-Beta Pruning.

**Key Classes:**
- `AIEngine`: Implements the AI player

**Key Methods:**
- `find_best_move()`: Returns the optimal move using Minimax
- `minimax(depth, is_maximizing, alpha, beta)`: Core Minimax with pruning algorithm

**Algorithm Explanation:**
- **Minimax**: Recursive algorithm that explores all possible game states
  - Maximizing player (AI) seeks highest score
  - Minimizing player (Human) seeks lowest score
  - Scores: AI wins (+10), Draw (0), Human wins (-10)
- **Alpha-Beta Pruning**: Reduces computation by skipping branches that won't affect the result
  - Alpha: Best value the maximizer can guarantee
  - Beta: Best value the minimizer can guarantee
  - If alpha >= beta, the branch can be pruned

### 3. `main.py`
Game controller and user interface.

**Key Classes:**
- `GameController`: Handles game flow and user interaction

**Features:**
- Welcome screen with instructions
- Input validation
- Game statistics tracking
- Play multiple rounds consecutively

## How to Run

```bash
python main.py
```

## Game Instructions

1. You play as **X**, the AI plays as **O**
2. Board positions are numbered 0-8:
   ```
    0 | 1 | 2
   -----------
    3 | 4 | 5
   -----------
    6 | 7 | 8
   ```
3. Enter your move when prompted (choose a position 0-8)
4. The AI will respond with its move
5. First to get three in a row wins!

## Algorithm Analysis

### Time Complexity
- **Without Pruning**: O(9!) = 362,880 node evaluations in worst case
- **With Alpha-Beta Pruning**: O(b^(d/2)) where b is branching factor, d is depth
- In practice: ~1,000-8,000 nodes evaluated per move (much faster!)

### Space Complexity
- O(d) where d is the maximum depth (9 for Tic-Tac-Toe)

### Game Tree Statistics
- Total possible game states: ~5,478
- Symmetric equivalents: ~765 unique states
- First move options: 9
- Average branching factor: ~6

## Key Concepts Demonstrated

### 1. **Game Theory**
- Zero-sum game (one player's gain is another's loss)
- Perfect information (both players can see all moves)
- Deterministic (no randomness)

### 2. **Search Algorithms**
- Depth-first search (DFS) exploration
- Recursive tree traversal
- Optimal move selection

### 3. **Optimization Techniques**
- Branch-and-bound (Alpha-Beta pruning)
- Move ordering for better pruning
- Memoization opportunities (for larger games)

### 4. **Scoring Systems**
- Depth-aware scoring (prefer quicker wins, slower losses)
- Terminal state evaluation
- Game tree evaluation

## Example Game Output

```
Starting a new game...

     |     |    
-----------
     |     |    
-----------
     |     |    

Your turn (X):
Enter your move (0-8): 4
AI is thinking...
AI placed O at position 0
(Nodes evaluated: 8465)

 O |   |   
-----------
   | X |   
-----------
   |   |   
```

## Performance Notes

- **First Move**: Takes longer (~8,000 nodes) as AI evaluates all possibilities
- **Mid-Game**: Faster (~500 nodes) as game tree shrinks
- **End-Game**: Very fast (~4 nodes) with few moves remaining

## Strategic Notes

The AI uses optimal play, meaning:
- If it's the first move, AI will tie or win in 100% of games
- The only way to draw is if both players play optimally
- You cannot beat this AI (it's unbeatable by design!)

## Possible Enhancements

1. **Move Ordering**: Pre-sort moves by heuristic value for better pruning
2. **Transposition Tables**: Cache previously evaluated positions
3. **Iterative Deepening**: Implement time-limited search
4. **Heuristic Evaluation**: Add evaluation function for incomplete games
5. **Opening Book**: Pre-compute and store optimal opening moves
6. **GUI Version**: Create graphical interface using tkinter or pygame
7. **Multi-level Difficulty**: Reduce search depth for easier AI variants

## Educational Value

This project teaches:
- Minimax algorithm implementation
- Alpha-Beta pruning optimization
- Recursive game tree exploration
- Python code organization and documentation
- Game AI fundamentals
- Algorithm performance analysis

## References

- Minimax Algorithm: https://en.wikipedia.org/wiki/Minimax
- Alpha-Beta Pruning: https://en.wikipedia.org/wiki/Alpha–beta_pruning
- Game Theory: https://en.wikipedia.org/wiki/Game_theory
- Tic-Tac-Toe: https://en.wikipedia.org/wiki/Tic-tac-toe

---

**Created**: January 2026  
**Python Version**: 3.6+  
**Dependencies**: None (uses only standard library)
