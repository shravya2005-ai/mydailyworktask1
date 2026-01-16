"""
AI Engine for Tic-Tac-Toe using Minimax Algorithm with Alpha-Beta Pruning.
This implements an unbeatable AI player using game theory principles.
"""

from tictactoe_game import TicTacToe


class AIEngine:
    """AI player using Minimax algorithm with Alpha-Beta Pruning."""
    
    def __init__(self, game):
        """
        Initialize the AI engine.
        
        Args:
            game (TicTacToe): The game instance
        """
        self.game = game
        self.max_depth = 9  # Maximum search depth
        self.nodes_evaluated = 0  # For statistics
    
    def find_best_move(self):
        """
        Find the best move for the AI using Minimax with Alpha-Beta Pruning.
        
        Returns:
            int: The best move position (0-8)
        """
        self.nodes_evaluated = 0
        best_score = float('-inf')
        best_move = None
        
        for move in self.game.get_available_moves():
            self.game.make_move(move, self.game.ai)
            score = self.minimax(0, False, float('-inf'), float('inf'))
            self.game.undo_move(move)
            
            if score > best_score:
                best_score = score
                best_move = move
        
        return best_move
    
    def minimax(self, depth, is_maximizing, alpha, beta):
        """
        Minimax algorithm with Alpha-Beta Pruning.
        
        Args:
            depth (int): Current depth in the game tree
            is_maximizing (bool): True if maximizing (AI's turn), False if minimizing (Human's turn)
            alpha (int): Alpha value for pruning
            beta (int): Beta value for pruning
            
        Returns:
            int: The score of the position
        """
        self.nodes_evaluated += 1
        
        # Terminal state evaluations
        if self.game.is_winner(self.game.ai):
            return 10 - depth  # Prefer quicker wins
        if self.game.is_winner(self.game.human):
            return depth - 10  # Prefer slower losses
        if self.game.is_draw():
            return 0
        
        # Depth-based pruning (limit search depth for efficiency)
        if depth >= self.max_depth:
            return 0
        
        if is_maximizing:  # AI's turn - maximize score
            max_score = float('-inf')
            for move in self.game.get_available_moves():
                self.game.make_move(move, self.game.ai)
                score = self.minimax(depth + 1, False, alpha, beta)
                self.game.undo_move(move)
                max_score = max(score, max_score)
                
                # Alpha-Beta Pruning
                alpha = max(alpha, score)
                if beta <= alpha:
                    break  # Prune remaining branches
            
            return max_score
        
        else:  # Human's turn - minimize score
            min_score = float('inf')
            for move in self.game.get_available_moves():
                self.game.make_move(move, self.game.human)
                score = self.minimax(depth + 1, True, alpha, beta)
                self.game.undo_move(move)
                min_score = min(score, min_score)
                
                # Alpha-Beta Pruning
                beta = min(beta, score)
                if beta <= alpha:
                    break  # Prune remaining branches
            
            return min_score
    
    def get_nodes_evaluated(self):
        """
        Get the number of nodes evaluated in the last search.
        
        Returns:
            int: Number of nodes evaluated
        """
        return self.nodes_evaluated
    
    def reset_stats(self):
        """Reset evaluation statistics."""
        self.nodes_evaluated = 0
