from AmoebaPlayGround.AmoebaAgent import ConsoleAgent
from AmoebaPlayGround.AmoebaView import ConsoleView
from AmoebaPlayGround.AmoebaTrainer import AmoebaTrainer
import numpy as np


consoleAgent = ConsoleAgent()
view = ConsoleView()
trainer = AmoebaTrainer(consoleAgent,consoleAgent)
trainer.train(1,map_size=(10,10),view= view)