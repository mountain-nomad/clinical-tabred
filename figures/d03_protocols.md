# Evaluation protocols

## Boxes
1. **2011-2012 (cycle G)** (position x=0, y=2.0)
2. **2013-2014 (cycle H)** (position x=4.8, y=2.0)
3. **2015-2016 (cycle I)** (position x=9.6, y=2.0)
4. **A2: 80% train, 20% in-period test, same model on 2015-16** (position x=0, y=0.2)
5. **B: all cycles pooled, 70/15/15 random (5 seeds)** (position x=4.8, y=0.2)
6. **5-fold CV on development data only** (position x=9.6, y=0.2)

## Arrows
- 2011-2012 (cycle G) → 2013-2014 (cycle H) (label: A: train → validate)
- 2013-2014 (cycle H) → 2015-2016 (cycle I) (label: A: → test (once))
