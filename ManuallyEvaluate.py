import AmoebaPlayGround.Amoeba as Amoeba
from AmoebaPlayGround.AmoebaView import GraphicalView
from AmoebaPlayGround.GameGroup import GameGroup
from AmoebaPlayGround.HandWrittenAgent import HandWrittenAgent

Amoeba.map_size = (10, 10)

graphical_view = GraphicalView(Amoeba.map_size)
hand_written_agent = HandWrittenAgent()
game = GameGroup(batch_size=1, x_agent=hand_written_agent, o_agent=graphical_view, view=graphical_view)
game.play_all_games()
