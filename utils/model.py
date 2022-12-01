from sentence_transformers import SentenceTransformer, LoggingHandler, util, models, evaluation, losses, InputExample
def loadModel(use_pre_trained_model,model_name,max_seq_length,pooling,*args, **kwargs):
    """_summary_

    Args:
        use_pre_trained_model (_type_): _description_
        model_name (_type_): _description_
        max_seq_length (_type_): _description_
        pooling (_type_): _description_

    Returns:
        _type_: _description_
    """
    # Load our embedding model
    if use_pre_trained_model:
        # logging.info("use pretrained SBERT model")
        model = SentenceTransformer(model_name)
        model.max_seq_length = max_seq_length
    else:
        # logging.info("Create new SBERT model")
        word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),pooling)
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    return model