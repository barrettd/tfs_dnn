// --------------------------------------------------------------------
//  DnnLayerConvolution.h
//
//  Created by Barrett Davis on 5/8/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
// --------------------------------------------------------------------
#ifndef DnnLayerConvolution_h
#define DnnLayerConvolution_h

#include "DnnLayer.h"

namespace tfs {
    
    class DnnLayerConvolution : public DnnLayer {   
    protected:
        unsigned long m_side;       // Prefer odd sizes, to give the filter a center.
        unsigned long m_stride;
        unsigned long m_pad;
        unsigned long m_filter_count;
        Matrix      **m_filter;
        
        void setup( const bool trainable = true );

    public:
        static const char *className( void );

        DnnLayerConvolution( DnnLayer *previousLayer,
                            unsigned long side,
                            unsigned long filters,
                            unsigned long stride = 2,
                            unsigned long pad = 0,
                            const bool trainable = true );
        virtual ~DnnLayerConvolution( void );
        
        virtual bool runForward(  void );
        virtual bool runBackprop( void );
    };
    
    
}   // namespace tfs

#endif /* DnnLayerConvolution_h */
