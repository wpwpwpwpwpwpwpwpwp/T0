from distutils.core import setup

setup(
    name='t0_framework',
    version='1.0',
    description='alpha with cta mode backend',
    author='f.x zhenw.xu',
    author_email='sdu.xuefu@gmail.com',
    url='',
    license='sino quant group',
    platforms='python 3.6',
    packages=['cpa',
              'cpa.calculator',
              'cpa.dataReader',
              'cpa.factorPool',
              'cpa.factorPool.example',
              'cpa.factorProcessor',
              'cpa.feed',
              'cpa.indicators',
              'cpa.indicators.panelIndicators',
              'cpa.indicators.seriesIndicators',
              'cpa.utils']
)

