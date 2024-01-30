from amisc.utils import get_logger


def test_logging():
    """Test logging and plotting utils"""
    logger = get_logger('tester', stdout=True)
    logger.info('Testing logger...')
