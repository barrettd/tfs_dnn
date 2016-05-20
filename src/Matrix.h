// --------------------------------------------------------------------
//  Matrix.hpp - 3D matrix <template> for DNN_NUMERIC and DNN_INTEGER
//
//  Created by Barrett Davis on 5/10/16.
//  Copyright © 2016 Tree Frog Software. All rights reserved.
// --------------------------------------------------------------------

#ifndef Matrix_h
#define Matrix_h

#include <cmath>        // sqrt()
#include <cstring>      // memset()
#include "Constants.h"
#include "Error.h"
#include "Utility.h"    // random()

namespace tfs {         // Tree Frog Software

    template <typename T> class TMatrix  {      // Possibly use Eigen matricies: https://eigen.tuxfamily.org/
    protected:
        unsigned long m_x;      // Width
        unsigned long m_y;      // Height
        unsigned long m_z;      // Depth
        unsigned long m_count;  // = x * w * h;  // Count of DNN_NUMERIC elements.
        unsigned long m_length; // = count * sizeof( T )
        T            *m_data;   // Pointer to data[] array:     = &data[0];
        T            *m_end;    // Pointer to end of the array: = &data[m_count];
        
        void allocate( void ) {
            if( m_count > 0 ) {
                m_length = m_count * sizeof( T );   // Length in bytes
                m_data   = new T[m_count];          // = &data[0];          Possibly random values.
                m_end    = m_data + m_count;        // = &m_data[m_count];
            }
        }

    public:
        TMatrix( const TMatrix &other, bool copyOther = false ) :       // Constructor
        m_x( other.m_x ), m_y( other.m_y ), m_z( other.m_z ), m_count( other.m_count ),
        m_length( 0 ), m_data( 0 ), m_end( 0 ) {
            allocate();
            if( copyOther ) {
                copy( other );
            }
        }

        TMatrix( const unsigned long xx, const unsigned long yy = 1, const unsigned long zz = 1 ):  // Constructor
        m_x( xx ), m_y( yy ), m_z( zz ), m_count( xx * yy * zz ),
        m_length( 0 ), m_data( 0 ), m_end( 0 ) {
            allocate();
        }
        
        ~TMatrix( void ) {      // Destructor
            delete[] m_data;
            m_data   = 0;
            m_end    = 0;
            m_count  = 0;
            m_length = 0;
        }
        
        inline unsigned long width(  void ) const { return m_x; }
        inline unsigned long height( void ) const { return m_y; }
        inline unsigned long depth(  void ) const { return m_z; }
        inline unsigned long count(  void ) const { return m_count;  }  // Count of elements.
        inline unsigned long length( void ) const { return m_length; }  // Length in bytes
        
        inline bool isEmpty( void ) const { return m_data == 0 || m_data >= m_end || m_count < 1 || m_length < 1; }
        
        inline       T* data( void ) {               return m_data; }   // Start of data array
        inline const T* end(  void ) const {         return m_end;  }   // End of data array
        inline const T* dataReadOnly( void ) const { return m_data; }   // Start of data array (const)
        
        void randomize( void ) {                // Fill matrix with small positive values
            // Weight normalization is done to equalize the output variance of every neuron,
            // otherwise neurons with a lot of incoming connections will have outputs with a larger variance
            if( isEmpty()) {
                log_error( "empty matrix" );
                return;
            }
            const double  scale = sqrt( 1.0 / m_count );
                  T *       ptr = m_data;
            const T * const end = m_end;
            while( ptr < end ) {
                *ptr++ = (T) random( scale );
            }
        }

        void zero( void ) {                         // Fill matrix with zeros
            if( m_data != 0 && m_length > 0 ) {
                memset( m_data, 0, m_length );      // Yields IEEE 0 for both integer and real valued variables.
            }
        }
        
        void copy( const TMatrix &matrix ) {        // Copy the contents another matrix.
            if( m_data != 0 && matrix.m_data != 0 && m_length > 0  && m_length == matrix.m_length ) {
                memcpy( m_data, matrix.m_data, m_length );
            }
        }
        
        bool equal( const TMatrix &matrix ) const { // Return true if matricies same dimension and contents.
            if( m_data == 0 || matrix.m_data == 0 || m_length < 1 || m_length != matrix.m_length ||
               m_x != matrix.m_x || m_y != matrix.m_y || m_z != matrix.m_z ) {
                return false;
            }
            return memcmp( m_data, matrix.m_data, m_length ) == 0;
        }

        inline T dot( const TMatrix &matrix ) const { // Calculate the dot product: scalar = lhs (dot) rhs;
            if( isEmpty()) {
                log_error( "empty matrix" );
                return 0;
            }
            if( matrix.m_count != m_count ) {
                log_error( "Matricies not the same size" );
                return 0;
            }
            const T *       lhs = m_data;
            const T * const end = m_end;
            const T *       rhs = matrix.m_data;
            T result = 0;
            while( lhs < end ) {
                result += *lhs++ * *rhs++;
            }
            return result;
        }
        
        inline T max( void ) const {                // Return maximum value.
            if( isEmpty()) {
                log_error( "empty matrix" );
                return 0;
            }
            const T *      data = m_data;
            const T * const end = m_end;
            
            T result = *data++;
            while( data < end ) {
                if( *data > result ) {
                    result = *data;
                }
                data++;
            }
            return result;
        }
        
        
    };
    
    typedef TMatrix< DNN_NUMERIC >  Matrix;     // Typically double
    typedef TMatrix< DNN_INTEGER > DMatrix;     // Typically long
    
}   // namespace tfs

#endif /* Matrix_h */
