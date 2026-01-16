"""
Main game loop and user interface for Tic-Tac-Toe.
This module handles user interaction and game flow.
"""

from tictactoe_game import TicTacToe
from ai_engine import AIEngine


class GameController:
    """Controls the game flow and user interaction."""
    
    def __init__(self):
        """Initialize the game controller."""
        self.game = TicTacToe()
        self.ai = AIEngine(self.game)
        self.human_wins = 0
        self.ai_wins = 0
        self.draws = 0
    
    def display_welcome(self):
        """Display welcome message and game instructions."""
        print("=" * 50)
        print("      WELCOME TO TIC-TAC-TOE AI")
        print("=" * 50)
        print("\nYou are X, AI is O")
        print("Board positions are numbered 0-8:")
        self.game.print_board_with_positions()
        print("Instructions:")
        print("- Enter a number (0-8) to place your mark")
        print("- Try to get three in a row (horizontal, vertical, or diagonal)")
        print("- The AI will use Minimax with Alpha-Beta Pruning")
        print("- Good luck!\n")
    
    def get_human_move(self):
        """
        Get a valid move from the human player.
        
        Returns:
            int: Valid board position (0-8)
        """
        while True:
            try:
                move = int(input("Enter your move (0-8): "))
                if move < 0 or move > 8:
                    print("Invalid! Please enter a number between 0 and 8.")
                    continue
                if not self.game.make_move(move, self.game.human):
                    print("That position is already taken!")
                    continue
                return move
            except ValueError:
                print("Invalid input! Please enter a number.")
    
    def play_round(self):
        """Play a single round of Tic-Tac-Toe."""
        print("\n" + "=" * 50)
        print("Starting a new game...")
        print("=" * 50)
        self.game = TicTacToe()
        self.ai = AIEngine(self.game)
        
        move_count = 0
        
        while not self.game.is_game_over():
            self.game.print_board()
            
            # Human's turn
            print("Your turn (X):")
            self.get_human_move()
            move_count += 1
            
            if self.game.is_game_over():
                break
            
            # AI's turn
            print("AI is thinking...")
            self.ai.reset_stats()
            ai_move = self.ai.find_best_move()
            self.game.make_move(ai_move, self.game.ai)
            print(f"AI placed O at position {ai_move}")
            print(f"(Nodes evaluated: {self.ai.get_nodes_evaluated()})")
            move_count += 1
        
        # Display final board
        self.game.print_board()
        self.display_result(move_count)
    
    def display_result(self, move_count):
        """
        Display the game result.
        
        Args:
            move_count (int): Total number of moves played
        """
        print("=" * 50)
        print("GAME OVER!")
        print("=" * 50)
        
        if self.game.is_winner(self.game.human):
            print("🎉 Congratulations! You won!")
            self.human_wins += 1
        elif self.game.is_winner(self.game.ai):
            print("🤖 AI wins! Well played.")
            self.ai_wins += 1
        else:
            print("It's a draw!")
            self.draws += 1
        
        print(f"\nTotal moves: {move_count}")
        self.display_stats()
    
    def display_stats(self):
        """Display overall game statistics."""
        total_games = self.human_wins + self.ai_wins + self.draws
        print(f"\n{'=' * 50}")
        print(f"Statistics: {total_games} games played")
        print(f"Human Wins: {self.human_wins}")
        print(f"AI Wins: {self.ai_wins}")
        print(f"Draws: {self.draws}")
        if total_games > 0:
            print(f"Win Rate: {(self.human_wins/total_games)*100:.1f}%")
        print(f"{'=' * 50}\n")
    
    def play_game(self):
        """Main game loop - allows player to play multiple rounds."""
        self.display_welcome()
        
        while True:
            self.play_round()
            
            while True:
                choice = input("\nPlay another game? (y/n): ").lower().strip()
                if choice in ['y', 'yes']:
                    break
                elif choice in ['n', 'no']:
                    print("\nThanks for playing Tic-Tac-Toe AI!")
                    self.display_stats()
                    return
                else:
                    print("Please enter 'y' for yes or 'n' for no.")


def main():
    """Entry point for the game."""
    controller = GameController()
    controller.play_game()


if __name__ == "__main__":
    main()
