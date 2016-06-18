// --------------------------------------------------------------------
//  DnnLayerPool.h
//
//  Created by Barrett Davis on 5/8/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
// --------------------------------------------------------------------
#ifndef DnnLayerPool_h
#define DnnLayerPool_h

#include "DnnLayer.h"

namespace tfs {
    
    class DnnLayerPool : public DnnLayer {
    protected:
        unsigned long  m_side;
        unsigned long  m_stride;
        unsigned long  m_pad;
        unsigned long  m_switch_count;
        unsigned long *m_switch;  // Contains index to max max values
        
        void setup( const bool trainable = true );

    public:
        DnnLayerPool( DnnLayer *previousLayer,
                     unsigned long side,            // Size of the square side
                     unsigned long stride = 2,
                     unsigned long pad    = 0,
                     const bool trainable = true );
        virtual ~DnnLayerPool( void );
        
        unsigned long side(   void ) const;
        unsigned long stride( void ) const;
        unsigned long pad(    void ) const;
        unsigned long switchCount( void ) const;
        
        const unsigned long* switchsReadOnly( void ) const;
              unsigned long* switchs( void );
        
        
        virtual bool runForward(  void );
        virtual bool runBackprop( void );
        
    };
    
    
}   // namespace tfs

#endif /* DnnLayerPool_h */
