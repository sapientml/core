from typing import Optional

from sapientml.params import Config, String


class DefaultPreprocessConfig(Config):
    """Configuration arguments for DefaultPreprocess class.

    Attributes
    ----------
    use_pos_list : Optional[list[str]]
      List of parts-of-speech to be used during text analysis.
      This variable is used for japanese texts analysis.
      Select the part of speech below.
      "名詞", "動詞", "形容詞", "形容動詞", "副詞".
    use_word_stemming : bool default True
      Specify whether or not word stemming is used.
      This variable is used for japanese texts analysis.

    """

    use_pos_list: Optional[list[String]] = ["名詞", "動詞", "助動詞", "形容詞", "副詞"]
    use_word_stemming: bool = True
