"""
Tic-Tac-Toe Game with AI using Minimax Algorithm with Alpha-Beta Pruning
This module implements the game board logic and game state management.
"""

class TicTacToe:
    """Represents the Tic-Tac-Toe game board and game logic."""
    
    def __init__(self):
        """Initialize an empty Tic-Tac-Toe board."""
        self.board = [' ' for _ in range(9)]  # 3x3 board represented as a list
        self.human = 'X'  # Human player
        self.ai = 'O'     # AI player
    
    def print_board(self):
        """Display the current game board in a readable format."""
        print("\n")
        for i in range(3):
            print(f" {self.board[i*3]} | {self.board[i*3+1]} | {self.board[i*3+2]} ")
            if i < 2:
                print("-----------")
        print("\n")
    
    def print_board_with_positions(self):
        """Display the board with position numbers for reference."""
        print("\nBoard positions (0-8):")
        for i in range(3):
            print(f" {i*3} | {i*3+1} | {i*3+2} ")
            if i < 2:
                print("-----------")
        print("\n")
    
    def is_winner(self, player):
        """
        Check if the specified player has won.
        
        Args:
            player (str): 'X' for human or 'O' for AI
            
        Returns:
            bool: True if player has won, False otherwise
        """
        # Define all winning combinations
        win_conditions = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columns
            [0, 4, 8], [2, 4, 6]               # Diagonals
        ]
        
        for condition in win_conditions:
            if all(self.board[i] == player for i in condition):
                return True
        return False
    
    def is_draw(self):
        """
        Check if the game is a draw (board is full and no winner).
        
        Returns:
            bool: True if the game is a draw, False otherwise
        """
        return ' ' not in self.board and not self.is_winner(self.human) and not self.is_winner(self.ai)
    
    def is_game_over(self):
        """
        Check if the game is over (someone won or it's a draw).
        
        Returns:
            bool: True if game is over, False otherwise
        """
        return self.is_winner(self.human) or self.is_winner(self.ai) or self.is_draw()
    
    def get_available_moves(self):
        """
        Get all available moves (empty board positions).
        
        Returns:
            list: List of available position indices (0-8)
        """
        return [i for i in range(9) if self.board[i] == ' ']
    
    def make_move(self, position, player):
        """
        Make a move on the board.
        
        Args:
            position (int): Board position (0-8)
            player (str): 'X' for human or 'O' for AI
            
        Returns:
            bool: True if move was successful, False if position is occupied
        """
        if self.board[position] == ' ':
            self.board[position] = player
            return True
        return False
    
    def undo_move(self, position):
        """
        Undo a move on the board.
        
        Args:
            position (int): Board position to clear
        """
        self.board[position] = ' '
    
    def get_board_state(self):
        """
        Get a copy of the current board state.
        
        Returns:
            list: Copy of the current board
        """
        return self.board.copy()
    
    def set_board_state(self, state):
        """
        Set the board to a specific state.
        
        Args:
            state (list): New board state
        """
        self.board = state.copy()
