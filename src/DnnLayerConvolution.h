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
        
        void setup( const bool trainable = true );

    public:
        DnnLayerConvolution( DnnLayer    *previousLayer,
                            unsigned long side,             // Square filter size (prefer odd size)
                            unsigned long filterCount,
                            unsigned long stride    = 1,
                            unsigned long pad       = 0,
                            const bool    trainable = true );
        virtual ~DnnLayerConvolution( void );
        
        unsigned long side(        void ) const;
        unsigned long stride(      void ) const;
        unsigned long pad(         void ) const;
        unsigned long filterCount( void ) const;
        
        virtual bool runForward(  void );
        virtual bool runBackprop( void );
    };
    
    
}   // namespace tfs

#endif /* DnnLayerConvolution_h */
