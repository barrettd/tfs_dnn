// --------------------------------------------------------------------
//  DnnLayerDropout.h
//
//  Created by Barrett Davis on 5/8/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
// --------------------------------------------------------------------
#ifndef DnnLayerDropout_h
#define DnnLayerDropout_h

#include "DnnLayer.h"

namespace tfs {
    
    class DnnLayerDropout : public DnnLayer {
    protected:
        DNN_NUMERIC m_probability;
        bool       *m_dropped;
        void setup( const bool trainable = true );

    public:
        DnnLayerDropout( DnnLayer *previousLayer, DNN_NUMERIC probability = 0.5, const bool trainable = true );
        virtual ~DnnLayerDropout( void );
        
        DNN_NUMERIC probability( void ) const;
        
        virtual bool runForward(  void );
        virtual bool runPredict(  void );                       // Forward progagate when predicting
        virtual bool runBackprop( void );
    };
    
    
}   // namespace tfs

#endif /* DnnLayerDropout_h */
