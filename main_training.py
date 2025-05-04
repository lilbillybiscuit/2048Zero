from game_alt import GameRules
from zero2048 import *
from zero2048 import ZeroTrainer
from zeromodel import ZeroNetwork

def main():
    spawn_rates = {
        2: 0.9,
        4: 0.1
    }
    rules =GameRules(4, 4, spawn_rates, 2, 2)
    model = ZeroNetwork(4, 4, 20, 64, 10)
    player = ZeroPlayer(model, rules)
    trainer = ZeroTrainer(model, rules, player)
    trainer.train(10, 10, 10)

if __name__== "__main__":
    main()