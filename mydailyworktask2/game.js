/* ============================================
   TIC-TAC-TOE AI - GAME LOGIC
   ============================================ */

class TicTacToeGame {
    constructor() {
        this.board = Array(9).fill('');
        this.human = 'X';
        this.ai = 'O';
        this.currentPlayer = 'X';
        this.gameOver = false;
        this.humanWins = 0;
        this.aiWins = 0;
        this.draws = 0;
        this.lastNodesEvaluated = 0;
        this.initializeUI();
    }

    initializeUI() {
        this.cells = document.querySelectorAll('.cell');
        this.statusEl = document.getElementById('status');
        this.thinkingEl = document.getElementById('thinking');
        this.resetBtn = document.getElementById('resetBtn');
        this.infoBtn = document.getElementById('infoBtn');
        this.modal = document.getElementById('infoModal');
        this.closeBtn = document.getElementById('closeBtn');
        this.closeModalBtn = document.getElementById('closeModalBtn');
        this.humanScoreEl = document.getElementById('humanScore');
        this.aiScoreEl = document.getElementById('aiScore');
        this.drawScoreEl = document.getElementById('drawScore');
        this.nodesEvaluatedEl = document.getElementById('nodesEvaluated');

        this.cells.forEach(cell => {
            cell.addEventListener('click', (e) => this.handleCellClick(e));
        });

        this.resetBtn.addEventListener('click', () => this.resetGame());
        this.infoBtn.addEventListener('click', () => this.openModal());
        this.closeBtn.addEventListener('click', () => this.closeModal());
        this.closeModalBtn.addEventListener('click', () => this.closeModal());
        window.addEventListener('click', (e) => {
            if (e.target === this.modal) this.closeModal();
        });

        this.updateScores();
    }

    handleCellClick(e) {
        if (this.gameOver) return;
        
        const cell = e.target;
        const index = parseInt(cell.getAttribute('data-index'));

        if (this.board[index] !== '') return;

        this.board[index] = this.human;
        this.renderBoard();

        if (this.checkWinner(this.human)) {
            this.endGame('win');
            return;
        }

        if (this.isBoardFull()) {
            this.endGame('draw');
            return;
        }

        this.currentPlayer = this.ai;
        this.statusEl.style.display = 'none';
        this.thinkingEl.style.display = 'block';

        // AI move with slight delay for UX
        setTimeout(() => this.makeAIMove(), 500);
    }

    makeAIMove() {
        const aiMove = this.findBestMove();
        this.board[aiMove] = this.ai;
        this.lastNodesEvaluated = this.nodesEvaluated;
        this.nodesEvaluatedEl.textContent = this.lastNodesEvaluated;
        this.renderBoard();

        this.thinkingEl.style.display = 'none';
        this.statusEl.style.display = 'block';

        if (this.checkWinner(this.ai)) {
            this.endGame('lose');
            return;
        }

        if (this.isBoardFull()) {
            this.endGame('draw');
            return;
        }

        this.currentPlayer = this.human;
        this.statusEl.textContent = 'Your Turn';
    }

    findBestMove() {
        this.nodesEvaluated = 0;
        let bestScore = -Infinity;
        let bestMove = null;

        for (let i = 0; i < 9; i++) {
            if (this.board[i] === '') {
                this.board[i] = this.ai;
                const score = this.minimax(0, false, -Infinity, Infinity);
                this.board[i] = '';

                if (score > bestScore) {
                    bestScore = score;
                    bestMove = i;
                }
            }
        }

        return bestMove;
    }

    minimax(depth, isMaximizing, alpha, beta) {
        this.nodesEvaluated++;

        // Terminal state evaluations
        if (this.checkWinner(this.ai)) {
            return 10 - depth;
        }
        if (this.checkWinner(this.human)) {
            return depth - 10;
        }
        if (this.isBoardFull()) {
            return 0;
        }

        if (isMaximizing) {
            // AI's turn - maximize
            let maxScore = -Infinity;
            for (let i = 0; i < 9; i++) {
                if (this.board[i] === '') {
                    this.board[i] = this.ai;
                    const score = this.minimax(depth + 1, false, alpha, beta);
                    this.board[i] = '';
                    maxScore = Math.max(score, maxScore);

                    // Alpha-Beta Pruning
                    alpha = Math.max(alpha, score);
                    if (beta <= alpha) break;
                }
            }
            return maxScore;
        } else {
            // Human's turn - minimize
            let minScore = Infinity;
            for (let i = 0; i < 9; i++) {
                if (this.board[i] === '') {
                    this.board[i] = this.human;
                    const score = this.minimax(depth + 1, true, alpha, beta);
                    this.board[i] = '';
                    minScore = Math.min(score, minScore);

                    // Alpha-Beta Pruning
                    beta = Math.min(beta, score);
                    if (beta <= alpha) break;
                }
            }
            return minScore;
        }
    }

    checkWinner(player) {
        const winConditions = [
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
            [0, 3, 6],
            [1, 4, 7],
            [2, 5, 8],
            [0, 4, 8],
            [2, 4, 6]
        ];

        return winConditions.some(condition =>
            condition.every(index => this.board[index] === player)
        );
    }

    isBoardFull() {
        return this.board.every(cell => cell !== '');
    }

    renderBoard() {
        this.cells.forEach((cell, index) => {
            cell.textContent = this.board[index];
            cell.className = 'cell';
            
            if (this.board[index] === 'X') {
                cell.classList.add('x');
                cell.disabled = true;
            } else if (this.board[index] === 'O') {
                cell.classList.add('o');
                cell.disabled = true;
            } else {
                cell.disabled = false;
            }
        });
    }

    endGame(result) {
        this.gameOver = true;
        this.currentPlayer = null;

        if (result === 'win') {
            this.statusEl.textContent = '🎉 You Won! Congratulations!';
            this.humanWins++;
            this.highlightWinningCells(this.human);
        } else if (result === 'lose') {
            this.statusEl.textContent = '🤖 AI Wins! Well Played.';
            this.aiWins++;
            this.highlightWinningCells(this.ai);
        } else if (result === 'draw') {
            this.statusEl.textContent = "It's a Draw!";
            this.draws++;
        }

        this.updateScores();
        this.disableAllCells();
    }

    highlightWinningCells(player) {
        const winConditions = [
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
            [0, 3, 6],
            [1, 4, 7],
            [2, 5, 8],
            [0, 4, 8],
            [2, 4, 6]
        ];

        for (let condition of winConditions) {
            if (condition.every(index => this.board[index] === player)) {
                condition.forEach(index => {
                    this.cells[index].classList.add('winner');
                });
                break;
            }
        }
    }

    disableAllCells() {
        this.cells.forEach(cell => {
            cell.disabled = true;
        });
    }

    resetGame() {
        this.board = Array(9).fill('');
        this.currentPlayer = 'X';
        this.gameOver = false;
        this.statusEl.textContent = 'Your Turn';
        this.statusEl.style.display = 'block';
        this.thinkingEl.style.display = 'none';
        this.nodesEvaluatedEl.textContent = '0';
        this.renderBoard();
    }

    updateScores() {
        this.humanScoreEl.textContent = this.humanWins;
        this.aiScoreEl.textContent = this.aiWins;
        this.drawScoreEl.textContent = this.draws;
    }

    openModal() {
        this.modal.style.display = 'block';
    }

    closeModal() {
        this.modal.style.display = 'none';
    }
}

// Initialize game when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new TicTacToeGame();
});
